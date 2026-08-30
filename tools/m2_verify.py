#!/usr/bin/env python3
"""M2（実音声 smoke）のゴール 2〜4 を測って確かめる。

    uv run python tools/m2_verify.py --ckpt <ckpt.pt> --data <shard dir> --out <dir>

[実行計画](../doc/svc-plan.md) M2 は「WAV を聴いて F0 の追従・無声区間・長さが整合している」
ことを求めますが、聴くだけでは記録に残りません。ここでは同じことを**測って**残します。

- 長さ: 出力サンプル数が `T * hop` と一致するか
- F0 追従: 出力 WAV から RMVPE で F0 を取り直し、入力 F0 との相関と半音誤差
- 無声区間: 出力の有声判定が入力の V/UV とどれだけ一致するか
- 再現性: 同じ checkpoint・同じ seed で 2 回生成し、mel が bit 一致するか

**切り分け:** ground-truth mel をそのまま NHVSing に通した WAV も出します。音響モデルの誤差と
vocoder の誤差を分けるためで、M2 の打ち切り条件が求めているものです。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# CUDA の conv / matmul は既定で非決定的で、同じ入力でも実行ごとに ln-mel が
# max 8e-2 ずれる（実測）。M2 ゴール 4 は再現を求めるので、torch を import する前に
# 決定的モードを有効にする。CUBLAS_WORKSPACE_CONFIG は CUDA 初期化前に要る。
if os.environ.get("LEAPSINGER_NONDETERMINISTIC") != "1":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def voiced_stats(f0_a, uv_a, f0_b, uv_b):
    """2 つの F0 列を、両方が有声のフレームだけで比べる。"""
    both = (uv_a > 0.5) & (uv_b > 0.5) & (f0_a > 0) & (f0_b > 0)
    if both.sum() < 8:
        return {"frames": int(both.sum())}
    cents = 1200.0 * np.log2(f0_b[both] / f0_a[both])
    return {
        "frames": int(both.sum()),
        "corr": float(np.corrcoef(np.log2(f0_a[both]), np.log2(f0_b[both]))[0, 1]),
        "median_semitones": float(np.median(np.abs(cents)) / 100.0),
        "p90_semitones": float(np.percentile(np.abs(cents), 90) / 100.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="svc_shard.npz のあるディレクトリ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-steps", type=int, default=None)
    a = ap.parse_args()

    import torch

    from infer import infer_svc_mel, load_acoustic, load_vocoder, mel_to_wav
    from preprocess.f0_rmvpe import extract_f0_rmvpe

    deterministic = os.environ.get("LEAPSINGER_NONDETERMINISTIC") != "1"
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data = Path(a.data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    z = np.load(data / "svc_shard.npz")
    name = sorted(meta["phrases"])[0]

    f0_hz = z[f"{name}|f0_interp"]
    uv = z[f"{name}|uv"]
    gt_mel = z[f"{name}|mel"]
    item = {"content": z[f"{name}|content"],
            "f0_logf0": np.log2(np.maximum(f0_hz, 1.0)).astype(np.float32),
            "uv": uv.astype(np.float32),
            "loudness": z[f"{name}|loudness"]}

    model, cfg = load_acoustic(a.ckpt, device=a.device)
    # checkpoint の config は yaml のままではなく平坦化されている（train.py の ckpt_config）。
    steps = a.num_steps or int(cfg.get("infer_steps", 1))
    mel_a = infer_svc_mel(model, item, num_steps=steps, device=a.device)
    mel_b = infer_svc_mel(model, item, num_steps=steps, device=a.device)

    sr, hop = int(cfg["sample_rate"]), int(cfg["hop"])
    voc = load_vocoder(str(ROOT / "checkpoints" / "nhv_v3.onnx"))
    wav_pred = mel_to_wav(voc, mel_a, item["f0_logf0"], item["uv"])
    wav_gt = mel_to_wav(voc, gt_mel, item["f0_logf0"], item["uv"])
    sf.write(out / "m2_pred.wav", wav_pred, sr)
    sf.write(out / "m2_gt_mel.wav", wav_gt, sr)

    frames = gt_mel.shape[1]
    report = {
        "phrase": name, "frames": frames, "num_steps": steps,
        "goal2_wav_written": True,
        "goal3_length_ok": len(wav_pred) == frames * hop,
        "goal3_length": {"wav_samples": int(len(wav_pred)), "expected": frames * hop},
        "goal4_reproducible": bool(np.array_equal(mel_a, mel_b)),
        "goal4_deterministic_mode": deterministic,
        "goal4_max_diff": float(np.abs(mel_a - mel_b).max()),
        "mel_mae_vs_gt": float(np.abs(mel_a - gt_mel).mean()),
        "mel_range": [float(mel_a.min()), float(mel_a.max())],
        "wav_peak": float(np.abs(wav_pred).max()),
        "wav_finite": bool(np.isfinite(wav_pred).all()),
    }

    # 出力から F0 を取り直して、入力とどれだけ合っているかを測る（ゴール 3）。
    for tag, wav in (("pred", wav_pred), ("gt_mel", wav_gt)):
        f0_o, uv_o = extract_f0_rmvpe(wav, sr, hop, 65.0, 1100.0,
                                      device=a.device, interpolate=False)
        n = min(len(f0_o), frames)
        report[f"goal3_f0_{tag}"] = voiced_stats(f0_hz[:n], uv[:n],
                                                 np.asarray(f0_o)[:n], np.asarray(uv_o)[:n])
        report[f"goal3_uv_agree_{tag}"] = float(
            ((np.asarray(uv_o)[:n] > 0.5) == (uv[:n] > 0.5)).mean())

    (out / "m2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
