#!/usr/bin/env python3
"""推論の RTF と peak VRAM を測る（[実行計画](../doc/svc-plan.md) M5 ゴール 2）。

    uv run python tools/rtf.py --wav <vocal.wav> --ckpt <ckpt.pt> \
      --manifest <manifest.json> --spk-id 22 --device cpu --out out/rtf.json

**単一の RTF を出しません。** M5 ゴール 2 が「特徴抽出と vocoder を含むか除くかを併記」を
要求しているとおり、どこまでを数えるかで倍以上変わります。段を 4 つに分けます。

| 段 | 含む | 中身 |
|---|---|---|
| `content` | 特徴抽出 | ContentVec |
| `f0` | 特徴抽出 | RMVPE |
| `flow` | **acoustic** | LeapSVC 本体（rectified flow） |
| `vocoder` | ボコーダー | NHVSing |

`rtf_acoustic_only` は `flow` だけ、`rtf_total` は全部です。**「1-step だから速い」は前者、
「リアルタイムで使える」は後者**の話です。

**主張の制約（[先行研究・ライセンス](../doc/svc-prior-art-license.md) 6 節）:**
「リアルタイム」は**対象ハードウェアでの end-to-end 実測と連続動作**の後にのみ使えます。
`realtime_capable` は `rtf_total < 1` を機械的に見ているだけで、chunk 境界も I/O 遅延も
連続運転も見ていません。**この旗だけでリアルタイムと書かないこと。**
"""
from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# 段をどの区分に数えるか。**未知の段は黙って足しません**（含む / 除くの境界が曖昧になるため）。
FEATURE_STAGES = ("content", "f0", "loudness")
ACOUSTIC_STAGES = ("flow",)
VOCODER_STAGES = ("vocoder",)
KNOWN_STAGES = FEATURE_STAGES + ACOUSTIC_STAGES + VOCODER_STAGES


def measure_stage[T](fn: Callable[[], T], *, repeats: int = 1) -> tuple[T, float]:
    """`fn` を `repeats` 回実行し、（最後の戻り値, 所要秒の中央値）を返す。

    **初回だけを測りません。** 重みの読み込みやカーネルの初回コンパイルを含むためです。
    """
    if repeats < 1:
        raise ValueError(f"repeats は 1 以上（{repeats} が来ました）")
    times: list[float] = []
    out: Any = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(statistics.median(times))


def rtf_from_stages(stage_seconds: dict[str, float], *, audio_seconds: float) -> dict[str, Any]:
    """段ごとの所要から RTF を組み立てる。RTF = 計算時間 / 音声長。"""
    if audio_seconds <= 0:
        raise ValueError(f"音声長が {audio_seconds} 秒です。RTF を定義できません")
    unknown = sorted(set(stage_seconds) - set(KNOWN_STAGES))
    if unknown:
        raise ValueError(
            f"未知の段 {unknown} があります。{list(KNOWN_STAGES)} のどれかに分類してください"
            "（含む / 除くの境界が曖昧になるため、黙って合計には入れません）")

    def total(keys):
        return sum(float(stage_seconds.get(k, 0.0)) for k in keys)

    all_sec = total(stage_seconds)
    return {
        "audio_seconds": float(audio_seconds),
        "stage_seconds": {k: float(v) for k, v in stage_seconds.items()},
        "stage_rtf": {k: float(v) / audio_seconds for k, v in stage_seconds.items()},
        "rtf_total": all_sec / audio_seconds,
        "rtf_acoustic_only": total(ACOUSTIC_STAGES) / audio_seconds,
        "rtf_features": total(FEATURE_STAGES) / audio_seconds,
        "rtf_vocoder": total(VOCODER_STAGES) / audio_seconds,
        "realtime_capable": bool(all_sec / audio_seconds < 1.0),
        "caveat": ("realtime_capable は rtf_total < 1 を見ているだけ。chunk 境界・audio I/O・"
                   "連続運転を測るまで「リアルタイム」とは書かないこと"),
    }


def _peak_vram_mb(device: str) -> float | None:
    """CUDA の peak VRAM（MB）。CPU なら None。"""
    if not str(device).startswith("cuda"):
        return None
    import torch
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)


def main() -> int:
    import argparse
    import json

    import numpy as np
    import soundfile as sf
    import torch

    from infer import infer_svc_mel, load_acoustic, load_vocoder, mel_to_wav
    from leapsinger.config import MelSpec
    from preprocess.svc.encoders import ContentVecEncoder, RmvpeF0
    from preprocess.svc.extract import _resample, extract_phrase
    from preprocess.svc.shard import features_to_item

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True, help="計測に使うボーカル WAV")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True, help="target 話者の shard の manifest.json")
    ap.add_argument("--spk-id", type=int, default=22)
    ap.add_argument("--num-steps", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=10.0, help="計測に使う長さ")
    ap.add_argument("--repeats", type=int, default=3, help="各段の反復回数（中央値を取る）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--content-model", default="lengyue233/content-vec-best")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vocoder", default="checkpoints/nhv_v3.onnx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mel = MelSpec()
    w, sr = sf.read(a.wav, dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    w = _resample(np.ascontiguousarray(w, dtype=np.float32), sr, mel.sr)
    w = w[: int(a.seconds * mel.sr)]
    audio_seconds = len(w) / mel.sr
    print(f"[rtf] {audio_seconds:.2f}s / device={a.device} / repeats={a.repeats}")

    enc = ContentVecEncoder(a.content_model, layer=a.layer, device=a.device)
    f0x = RmvpeF0(device=a.device)
    model, _cfg = load_acoustic(a.ckpt, device=a.device)   # (model, config) を返す
    voc = load_vocoder(a.vocoder)
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))

    stages: dict[str, float] = {}
    # **段ごとに測る。** 呼び方は tools/svc_convert.py の変換ループと同じにする
    # （ContentVec は 16 kHz、RMVPE は mel.sr と hop を取る）。
    w16 = _resample(w, mel.sr, 16000)
    _, stages["content"] = measure_stage(lambda: enc(w16, 16000), repeats=a.repeats)
    _, stages["f0"] = measure_stage(lambda: f0x(w, mel.sr, mel.hop), repeats=a.repeats)

    feats = extract_phrase(w, mel.sr, content_encoder=enc, f0_extract=f0x, mel=mel)
    item = features_to_item(feats, manifest)
    item["spk_id"] = int(a.spk_id)
    logf0 = np.log2(np.maximum(feats["f0_hz"], 1.0))
    _, stages["flow"] = measure_stage(
        lambda: infer_svc_mel(model, item, num_steps=a.num_steps, device=a.device),
        repeats=a.repeats)
    pred = infer_svc_mel(model, item, num_steps=a.num_steps, device=a.device)
    _, stages["vocoder"] = measure_stage(
        lambda: mel_to_wav(voc, pred, logf0, feats["uv"]), repeats=a.repeats)

    rep = rtf_from_stages(stages, audio_seconds=audio_seconds)
    rep["peak_vram_mb"] = _peak_vram_mb(a.device)
    rep["manifest"] = {"device": a.device, "num_steps": a.num_steps, "ckpt": a.ckpt,
                       "repeats": a.repeats, "content_model": a.content_model,
                       "vocoder": a.vocoder}

    print("\n=== RTF（段ごと）===")
    for k in sorted(rep["stage_rtf"]):
        print(f"  {k:9s} {rep['stage_seconds'][k]:7.3f}s  RTF {rep['stage_rtf'][k]:6.3f}")
    print(f"\n  acoustic のみ : {rep['rtf_acoustic_only']:.3f}")
    print(f"  特徴抽出      : {rep['rtf_features']:.3f}")
    print(f"  ボコーダー    : {rep['rtf_vocoder']:.3f}")
    print(f"  **合計**      : {rep['rtf_total']:.3f}")
    if rep["peak_vram_mb"] is not None:
        print(f"  peak VRAM     : {rep['peak_vram_mb']:.1f} MB")
    print(f"\n  ** {rep['caveat']} **")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
