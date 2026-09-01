#!/usr/bin/env python3
"""onset のずれを測る（[実行計画](../doc/svc-plan.md) M5 ゴール 2 の timing）。

変換が source の発音タイミングを保っているかを見ます。**source を基準**にし、変換後の onset が
どれだけ動いたかを測ります。

**「ずれの平均」だけでは読めません。** onset が消えた・増えた場合はペアが作れず、平均からは
黙って抜け落ちます。子音が落ちて onset ごと消えるのは実際に起こる失敗
（[評価計画](../doc/svc-evaluation.md) の failure taxonomy の `content` / `timing`）なので、
**対応が付いた割合**（`matched_ratio`）を必ず併せて読みます。

**符号つきのずれも返します。** 遅れと進みが打ち消し合うと「ずれていない」ように見えるためです
（明るさを符号つき平均で評価して誤った前例があります）。

**分解能は hop です。** 既定の hop 256 / 44.1 kHz なら **5.8 ms** で、これより細かいずれは
測れません。実測（M4 の ft10000、6 clip）でずれの中央値はちょうど 5.8 ms = 1 フレームでした。
**つまりずれは測定限界以下で、読むべき信号は `matched_ratio`（実測 69.1%）のほうです。**
「ずれが小さい＝timing が保たれている」と読まないこと。onset の 3 割は対応が付いていません。
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def detect_onsets(wav: np.ndarray, sr: int, *, hop: int = 256) -> list[float]:
    """波形から onset の時刻（秒）を返す。"""
    import librosa

    x = np.ascontiguousarray(np.asarray(wav, dtype=np.float32).ravel())
    if not np.any(np.isfinite(x)) or float(np.max(np.abs(x))) <= 0.0:
        return []
    times = librosa.onset.onset_detect(y=x, sr=int(sr), hop_length=int(hop), units="time")
    return [float(t) for t in times]


def hop_resolution_seconds(sr: int, hop: int) -> float:
    """onset 検出の分解能（秒）。**これより細かいずれは測れません。**"""
    return float(hop) / float(sr)


def onset_deviation(source: Sequence[float], converted: Sequence[float], *,
                    tol: float = 0.05) -> dict[str, Any]:
    """source の各 onset に、`tol` 秒以内で最も近い変換後の onset を 1 つだけ対応させる。

    **1 つの onset を複数に対応させません。** 対応させるとずれが小さく見えるうえ、
    本数（消えた / 増えた）が合わなくなります。

    返す値:

    | key | 意味 |
    |---|---|
    | `median_abs_dev` | 対応が付いた組のずれの絶対値の中央値（秒）。**対応が無ければ None** |
    | `median_signed_dev` | 同じく符号つき。正なら変換後のほうが遅い |
    | `matched` / `missed` / `spurious` | 対応した数 / source 側で消えた数 / 変換側で増えた数 |
    | `matched_ratio` | `matched / len(source)`。**source に onset が無ければ None** |
    """
    src = [float(t) for t in source]
    cnv = [float(t) for t in converted]
    used: set[int] = set()
    devs: list[float] = []

    for s in src:
        best, best_d = None, None
        for j, c in enumerate(cnv):
            if j in used:
                continue
            d = c - s
            if abs(d) <= tol and (best_d is None or abs(d) < abs(best_d)):
                best, best_d = j, d
        if best is not None:
            used.add(best)
            devs.append(best_d)

    matched = len(devs)
    out: dict[str, Any] = {
        "matched": matched,
        "missed": len(src) - matched,
        "spurious": len(cnv) - matched,
        "n_source": len(src),
        "n_converted": len(cnv),
        "tol_seconds": float(tol),
        "median_abs_dev": float(np.median(np.abs(devs))) if devs else None,
        "median_signed_dev": float(np.median(devs)) if devs else None,
        "matched_ratio": (matched / len(src)) if src else None,
    }
    return out


def timing_report(source_wav: np.ndarray, converted_wav: np.ndarray, sr: int, *,
                  tol: float = 0.05, hop: int = 256) -> dict[str, Any]:
    """波形 2 本から onset を取り、ずれを返す。"""
    rep = onset_deviation(detect_onsets(source_wav, sr, hop=hop),
                          detect_onsets(converted_wav, sr, hop=hop), tol=tol)
    rep["seconds"] = {"source": len(source_wav) / sr, "converted": len(converted_wav) / sr}
    # **分解能を必ず添える。** 実測でずれの中央値がちょうど 1 フレームだった。
    # 読む人がそれを「ほぼずれていない」と取り違えないようにする。
    rep["resolution_seconds"] = hop_resolution_seconds(sr, hop)
    return rep


def main() -> int:
    import argparse
    import json

    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="`*_source.wav` と `*_converted.wav` を含むディレクトリ")
    ap.add_argument("--tol", type=float, default=0.05, help="対応とみなすずれの上限（秒）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.dir)
    pairs = []
    for src in sorted(root.glob("*_source.wav")):
        cnv = src.with_name(src.name.replace("_source.wav", "_converted.wav"))
        if cnv.exists():
            pairs.append((src, cnv))
    if not pairs:
        sys.exit(f"{a.dir} に *_source.wav / *_converted.wav の組が見つかりません")

    rows = []
    for src, cnv in pairs:
        s, sr = sf.read(src, dtype="float32", always_2d=False)
        c, sr2 = sf.read(cnv, dtype="float32", always_2d=False)
        if sr != sr2:
            sys.exit(f"sample rate が違います: {src} {sr} vs {cnv} {sr2}")
        r = timing_report(s, c, sr, tol=a.tol)
        r["file"] = src.name
        rows.append(r)
        dev = "n/a" if r["median_abs_dev"] is None else f"{r['median_abs_dev'] * 1000:5.1f} ms"
        ratio = "n/a" if r["matched_ratio"] is None else f"{r['matched_ratio'] * 100:5.1f}%"
        print(f"  {src.name:44s} ずれ {dev}  対応 {ratio}  "
              f"(消失 {r['missed']} / 余分 {r['spurious']})")

    ok = [r for r in rows if r["median_abs_dev"] is not None]
    summary = {
        "n_clips": len(rows),
        "median_abs_dev": float(np.median([r["median_abs_dev"] for r in ok])) if ok else None,
        "matched_ratio": float(np.mean([r["matched_ratio"] for r in rows
                                        if r["matched_ratio"] is not None])) if ok else None,
        "tol_seconds": float(a.tol),
    }
    print("\n=== timing ===")
    if summary["median_abs_dev"] is None:
        print("  onset が検出できませんでした（無声のみのクリップ？）")
    else:
        res = rows[0]["resolution_seconds"]
        at_floor = " ** 分解能と同じ。ずれは測定限界以下 **" if summary[
            "median_abs_dev"] <= res * 1.5 else ""
        print(f"  ずれの中央値 : {summary['median_abs_dev'] * 1000:.1f} ms"
              f"（分解能 {res * 1000:.1f} ms）{at_floor}")
        print(f"  対応した割合 : {summary['matched_ratio'] * 100:.1f}%"
              " <- onset の消失・増加はここに出る")
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
