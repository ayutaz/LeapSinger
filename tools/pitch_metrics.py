#!/usr/bin/env python3
"""F0 追従と V/UV を、変換済みの WAV から測る（[実行計画](../doc/svc-plan.md) M5 ゴール 2）。

    uv run python tools/pitch_metrics.py --dir out/m5/leapsvc_gpu \
      --testset out/m5/testset.json --device cuda --out out/m5/metrics_leapsvc/pitch.json

**`m3_verify.py` は使えません。** あれは自分で変換してしまうので、**既に作った test set を
両システムで同じように測る**ことができません。ここでは `<name>_source.wav` と
`<name>_converted.wav` から F0 と V/UV を取り、比べます。

**移調ぶんは誤差として数えません。** 男性 source は +12 半音で変換しているので、
その分を引いてから比べます（引かないと「12 半音ずれている」と出て意味がなくなります）。

**無声フレームは除きます。** 無声区間の F0 に意味はなく、混ぜると誤差が壊れます。
**有声フレームが 1 つも無ければ `None`** を返します（0 を返すと「完璧に一致」と読めます）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def pitch_report(source: tuple[np.ndarray, np.ndarray],
                 converted: tuple[np.ndarray, np.ndarray], *,
                 transpose: float = 0.0) -> dict[str, Any]:
    """(f0_hz, uv) の組を 2 つ比べる。

    | key | 意味 |
    |---|---|
    | `f0_corr` | 有声フレームの log2 F0 の相関 |
    | `median_abs_semitones` | 半音差の絶対値の中央値（**移調ぶんを引いた後**） |
    | `uv_agree` | V/UV が一致したフレームの割合 |
    """
    f0a, uva = (np.asarray(x, dtype=np.float64).ravel() for x in source)
    f0b, uvb = (np.asarray(x, dtype=np.float64).ravel() for x in converted)
    # **黙って伸ばさず、短いほうへ揃える。** 系ごとに端数フレームが違う。
    n = min(len(f0a), len(f0b), len(uva), len(uvb))
    f0a, f0b, uva, uvb = f0a[:n], f0b[:n], uva[:n], uvb[:n]

    out: dict[str, Any] = {
        "n_frames": int(n),
        "uv_agree": float(np.mean((uva > 0.5) == (uvb > 0.5))) if n else None,
        "transpose": float(transpose),
        "f0_corr": None, "median_abs_semitones": None, "n_voiced": 0,
    }
    both = (uva > 0.5) & (uvb > 0.5) & (f0a > 0) & (f0b > 0)
    out["n_voiced"] = int(both.sum())
    if out["n_voiced"] < 2:
        return out

    la, lb = np.log2(f0a[both]), np.log2(f0b[both])
    lb = lb - float(transpose) / 12.0          # 移調ぶんを引いてから比べる
    out["median_abs_semitones"] = float(np.median(np.abs(lb - la) * 12.0))
    if la.std() > 1e-9 and lb.std() > 1e-9:
        out["f0_corr"] = float(np.corrcoef(la, lb)[0, 1])
    elif np.allclose(la, lb):
        out["f0_corr"] = 1.0
    return out


def main() -> int:
    import argparse
    import json

    import soundfile as sf

    from leapsinger.config import MelSpec
    from preprocess.svc.encoders import RmvpeF0
    from preprocess.svc.extract import _resample

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="`*_source.wav` と `*_converted.wav` を含むディレクトリ")
    ap.add_argument("--testset", default=None,
                    help="testset.json。**移調量をここから読む**（渡さないと 0 として扱う）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tp_of: dict[str, float] = {}
    if a.testset:
        ts = json.loads(Path(a.testset).read_text(encoding="utf-8"))
        for kind in ("unseen", "holdout"):
            for i, c in enumerate(ts[kind]):
                tp_of[f"{kind}{i:02d}"] = float(c.get("transpose", 0.0))

    mel = MelSpec()
    f0x = RmvpeF0(device=a.device)

    def f0_of(path: Path):
        w, sr = sf.read(path, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        w = _resample(np.ascontiguousarray(w, dtype=np.float32), sr, mel.sr)
        return f0x(w, mel.sr, mel.hop)

    rows = []
    for src in sorted(Path(a.dir).glob("*_source.wav")):
        stem = src.name[: -len("_source.wav")]
        cnv = src.with_name(f"{stem}_converted.wav")
        if not cnv.exists():
            continue
        tag = stem.split("__")[-1]
        r = pitch_report(f0_of(src), f0_of(cnv), transpose=tp_of.get(tag, 0.0))
        r["file"] = stem
        r["tag"] = tag
        rows.append(r)
        semi = "n/a" if r["median_abs_semitones"] is None else f"{r['median_abs_semitones']:5.2f}"
        corr = "n/a" if r["f0_corr"] is None else f"{r['f0_corr']:.4f}"
        print(f"  {tag:10s} corr {corr}  半音差 {semi}  V/UV {r['uv_agree'] * 100:5.1f}%")

    ok = [r for r in rows if r["f0_corr"] is not None]
    summary = {
        "n_clips": len(rows), "n_usable": len(ok),
        "f0_corr": float(np.median([r["f0_corr"] for r in ok])) if ok else None,
        "median_abs_semitones": float(np.median(
            [r["median_abs_semitones"] for r in ok])) if ok else None,
        "uv_agree": float(np.mean([r["uv_agree"] for r in rows if r["uv_agree"] is not None]))
        if rows else None,
    }
    print("\n=== F0 と V/UV（中央値）===")
    if ok:
        print(f"  F0 相関   : {summary['f0_corr']:.4f}")
        print(f"  半音差    : {summary['median_abs_semitones']:.2f}（移調ぶんを引いた後）")
        print(f"  V/UV 一致 : {summary['uv_agree'] * 100:.1f}%")
    else:
        print("  有声フレームが足りず測れませんでした")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "clips": rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
