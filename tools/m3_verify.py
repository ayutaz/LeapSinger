#!/usr/bin/env python3
"""M3 ゴール 3 を測る: 学習に出ていない source singer を入れても内容が崩壊しないこと。

    uv run python tools/m3_verify.py --ckpt log/m3_base/ckpt_060000.pt \\
      --manifest data/ritsu/manifest.json --spk-id 4 \\
      --source .m0data/vocalset_wav --out out/m3 --device cuda

**測り方:** 「歌詞内容が崩壊しない」を聴かずに測るため、**content preservation** を使います。
source の ContentVec と、変換後の音声から取り直した ContentVec のフレームごとの cos 類似度です。
内容が保たれていれば高く、別人の別の歌になれば下がります。単独の数値には意味が無いので、
必ず 2 つの基準と並べます。

| 基準 | 意味 |
|---|---|
| **上限** | source の ground-truth mel をそのまま NHVSing に通した再合成。vocoder だけの劣化 |
| 変換 | source -> 学習した base model -> NHVSing |
| **下限** | 無関係な別クリップとの cos 類似度。内容が消えたときに落ちる先 |

あわせて F0 追従（source の音高に付いていくか）と V/UV 一致も測ります。SVC は source の
音高と内容を保ち、音色だけを target にするものなので、この 3 つが崩れていなければ
「内容が崩壊していない」と言えます。**音質や target らしさはここでは測りません**
（それは M4 / M5）。

**loudness の正規化について:** shard は話者ごとに自分の統計で正規化されています。未知の
source を推論するときは **target 話者の manifest の統計**を当てます（その話者の音量スケールで
モデルが学習しているため）。`--manifest` はそのための引数です。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("LEAPSINGER_NONDETERMINISTIC") != "1":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leapsinger.config import MelSpec                                    # noqa: E402
from tools.audio_metrics import band_profile                             # noqa: E402


def cos_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`[T, C]` 2 本のフレームごとの cos 類似度。長さは短いほうに合わせる。"""
    t = min(len(a), len(b))
    x, y = a[:t].astype(np.float64), b[:t].astype(np.float64)
    nx = np.linalg.norm(x, axis=1) + 1e-12
    ny = np.linalg.norm(y, axis=1) + 1e-12
    return ((x * y).sum(axis=1) / (nx * ny)).astype(np.float32)


def f0_stats(f0_a, uv_a, f0_b, uv_b) -> dict:
    both = (uv_a > 0.5) & (uv_b > 0.5) & (f0_a > 0) & (f0_b > 0)
    if int(both.sum()) < 16:
        return {"frames": int(both.sum())}
    cents = 1200.0 * np.log2(f0_b[both] / f0_a[both])
    return {"frames": int(both.sum()),
            "corr": float(np.corrcoef(np.log2(f0_a[both]), np.log2(f0_b[both]))[0, 1]),
            "median_semitones": float(np.median(np.abs(cents)) / 100.0),
            "p90_semitones": float(np.percentile(np.abs(cents), 90) / 100.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True, help="target 話者の shard の manifest.json")
    ap.add_argument("--spk-id", type=int, default=0, help="変換先の speaker id")
    ap.add_argument("--source", required=True, help="未知 source singer の WAV ディレクトリ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clips", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--min-voiced", type=float, default=0.3)
    ap.add_argument("--num-steps", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--content-model", default="lengyue233/content-vec-best")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vocoder", default="checkpoints/nhv_v3.onnx")
    ap.add_argument("--save-wav", type=int, default=3)
    a = ap.parse_args()

    import soundfile as sf
    import torch

    from infer import infer_svc_mel, load_acoustic, load_vocoder, mel_to_wav
    from leapsinger.mel import wav_to_mel_nhv
    from preprocess.svc.encoders import ContentVecEncoder, RmvpeF0
    from preprocess.svc.extract import _resample, extract_phrase
    from preprocess.svc.shard import features_to_item

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mel = MelSpec()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    model, cfg = load_acoustic(a.ckpt, device=a.device)
    vocoder = load_vocoder(str(ROOT / a.vocoder), sr=mel.sr)
    encoder = ContentVecEncoder(a.content_model, layer=a.layer, device=a.device)
    f0x = RmvpeF0(device=a.device)
    print(f"[m3] arch={cfg.get('arch')} n_speakers={cfg.get('n_speakers')} "
          f"spk_id={a.spk_id} steps={a.num_steps}", flush=True)

    src_root = Path(a.source)
    wavs = sorted(p for p in src_root.rglob("*.wav"))
    if not wavs:
        sys.exit(f"source が空です: {src_root}")

    def content_of(wav44: np.ndarray) -> np.ndarray:
        return np.asarray(encoder(_resample(wav44, mel.sr, 16000), 16000), dtype=np.float32)

    rows, saved, prev_content = [], 0, None
    for path in wavs:
        if len(rows) >= a.n_clips:
            break
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = _resample(np.ascontiguousarray(wav, dtype=np.float32), sr, mel.sr)
        n = int(a.seconds * mel.sr)
        if len(wav) < n:
            continue
        seg = np.ascontiguousarray(wav[(len(wav) - n) // 2:(len(wav) - n) // 2 + n])
        # **音量を触らない。** 学習は生の音量のまま特徴を取るので、ここで peak 正規化すると
        # loudness 条件が学習分布から 1.2 sigma ずれ、出力の高域が削れる（実測）。

        feats = extract_phrase(seg, mel.sr, content_encoder=encoder, f0_extract=f0x, mel=mel)
        if float(np.mean(feats["uv"] > 0.5)) < a.min_voiced:
            continue

        item = features_to_item(feats, manifest)
        item["spk_id"] = int(a.spk_id)
        pred_mel = infer_svc_mel(model, item, num_steps=a.num_steps, device=a.device)
        f0_hz, uv = feats["f0_hz"], feats["uv"]
        conv = mel_to_wav(vocoder, pred_mel, np.log2(np.maximum(f0_hz, 1.0)), uv)
        resyn = mel_to_wav(vocoder, feats["mel"], np.log2(np.maximum(f0_hz, 1.0)), uv)

        src_c = feats["content"]
        row = {"file": str(path.relative_to(ROOT) if ROOT in path.parents else path),
               "voiced_ratio": float(np.mean(uv > 0.5)),
               "content_cos_converted": float(np.mean(cos_per_frame(src_c, content_of(conv)))),
               "content_cos_resynth": float(np.mean(cos_per_frame(src_c, content_of(resyn))))}
        if prev_content is not None:            # 下限: 無関係なクリップとの類似度
            row["content_cos_unrelated"] = float(np.mean(cos_per_frame(src_c, prev_content)))
        prev_content = src_c

        f0_c, uv_c = f0x(np.ascontiguousarray(conv), mel.sr, mel.hop)
        row["f0_converted"] = f0_stats(f0_hz, uv, np.asarray(f0_c, np.float32),
                                       np.asarray(uv_c, np.float32))
        row["uv_agree_converted"] = float(np.mean((uv > 0.5) == (np.asarray(uv_c) > 0.5)))
        # **音の明るさ。** content cos も F0 も高域の欠落を検知しない（実測で centroid が
        # 620 -> 368 Hz に落ちても cos は 0.8217 -> 0.8096 しか動かなかった）ので別軸で測る。
        for tag, w in (("source", seg), ("resynth", resyn), ("converted", conv)):
            bp = band_profile(np.asarray(w, dtype=np.float32), mel.sr)
            if bp:
                row[f"centroid_{tag}"] = bp["centroid_hz"]
                row[f"bands_{tag}"] = bp["bands"]
        rows.append(row)
        if saved < a.save_wav:
            _p = float(np.abs(seg).max())        # 試聴用だけ揃える（特徴には使わない）
            sf.write(out_dir / f"m3_{saved}_source.wav",
                     (seg / _p * 0.95).astype(np.float32) if _p > 1e-6 else seg, mel.sr)
            sf.write(out_dir / f"m3_{saved}_converted.wav", conv, mel.sr)
            saved += 1
        print(f"  {path.name[:36]:<36} cos {row['content_cos_converted']:.3f} "
              f"(上限 {row['content_cos_resynth']:.3f})", flush=True)

    if not rows:
        sys.exit("測れるクリップがありませんでした（--seconds か --min-voiced を見直す）")

    def agg(key, sub=None):
        vals = [(r[key][sub] if sub else r[key]) for r in rows
                if key in r and (sub is None or sub in r[key])]
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                "n": len(vals)} if vals else None

    report = {"ckpt": a.ckpt, "spk_id": a.spk_id, "num_steps": a.num_steps,
              "source": str(src_root), "n_clips": len(rows), "seconds": a.seconds,
              "content_cos_converted": agg("content_cos_converted"),
              "content_cos_resynth_ceiling": agg("content_cos_resynth"),
              "content_cos_unrelated_floor": agg("content_cos_unrelated"),
              "f0_corr": agg("f0_converted", "corr"),
              "f0_median_semitones": agg("f0_converted", "median_semitones"),
              "uv_agree": agg("uv_agree_converted"),
              "centroid_source": agg("centroid_source"),
              "centroid_resynth_ceiling": agg("centroid_resynth"),
              "centroid_converted": agg("centroid_converted"), "clips": rows}
    (out_dir / "m3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    c, ceil, floor = (report["content_cos_converted"], report["content_cos_resynth_ceiling"],
                      report["content_cos_unrelated_floor"])
    print("\n=== M3 ゴール 3: 未知 source の内容保持 ===")
    print(f"  変換後の content cos : {c['mean']:.4f}  (n={c['n']})")
    print(f"  上限（GT mel 再合成） : {ceil['mean']:.4f}")
    if floor:
        print(f"  下限（無関係クリップ）: {floor['mean']:.4f}")
        span = ceil["mean"] - floor["mean"]
        if span > 1e-6:
            print(f"  -> 下限からの回復率  : {(c['mean'] - floor['mean']) / span * 100:.1f}%")
    cs, cc, cv = (report["centroid_source"], report["centroid_resynth_ceiling"],
                  report["centroid_converted"])
    if cs and cc and cv:
        print(f"  音の明るさ（spectral centroid）: source {cs['mean']:.0f} Hz / "
              f"上限 {cc['mean']:.0f} Hz / 変換 {cv['mean']:.0f} Hz")
        print("  上限より大きく下なら高域が落ちている（content cos では検知できない）")
    print(f"  F0 相関 {report['f0_corr']['mean']:.4f} / "
          f"中央値 {report['f0_median_semitones']['median']:.3f} 半音 / "
          f"V/UV {report['uv_agree']['mean']:.3f}")
    print(f"\n-> {out_dir / 'm3_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
