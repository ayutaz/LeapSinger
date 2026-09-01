#!/usr/bin/env python3
"""failure sample を分類して残す（[実行計画](../doc/svc-plan.md) M5 ゴール 4）。

    uv run python tools/failure_taxonomy.py --metrics out/m5/metrics_leapsvc \
      --dir out/m5/leapsvc_gpu --out out/m5/metrics_leapsvc/failures.json

**failure を除外しません。** 悪い clip を捨てれば平均は良くなりますが、**何が壊れるのかが
分からなくなります**（[評価計画](../doc/svc-evaluation.md) 7 節の failure taxonomy）。

**機械的に判定できるものだけを扱います。** 「こもっている」「不自然」といった聴感は
blind test の担当で、ここでは扱いません。判定にはすでに測った数値を使います。

| 分類 | 判定 |
|---|---|
| `silence` | 出力の peak がほぼ 0 |
| `nonfinite` | NaN / Inf を含む |
| `pitch` | 半音差の中央値が大きい（**移調ぶんを引いた後**） |
| `voicing` | V/UV の一致率が低い |
| `timing` | onset の対応率が低い |
| `content` | CER が上限より大きく悪化 |
| `timbre` | 明るさが上限から大きく外れる（**両方向**。符号つきだと打ち消し合う） |

**測れなかった軸は「失敗」に数えません**（CER は 8 clip で判別不能でした）。
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# 判定の境界。**測定の前に決めた値ではなく、明らかな破綻だけを拾う保守的な値**にしてある。
# ここを厳しくすると「失敗」が増えて平均が良く見えるので、緩めに置いて全部残す。
SILENCE_PEAK = 1e-4
PITCH_SEMITONES = 1.0          # 1 半音を超えるずれ（移調ぶんを引いた後）
VOICING_AGREE = 0.85
TIMING_MATCHED = 0.40
CER_EXCESS = 0.50
CENTROID_LOW, CENTROID_HIGH = 0.7, 1.5


def classify(clip: dict[str, Any]) -> list[str]:
    """1 clip を分類する。**当てはまらなければ空リスト。**"""
    out: list[str] = []
    if not clip.get("finite", True):
        out.append("nonfinite")
    peak = clip.get("peak")
    if peak is not None and float(peak) < SILENCE_PEAK:
        out.append("silence")
    semi = clip.get("median_abs_semitones")
    if semi is not None and float(semi) > PITCH_SEMITONES:
        out.append("pitch")
    uv = clip.get("uv_agree")
    if uv is not None and float(uv) < VOICING_AGREE:
        out.append("voicing")
    matched = clip.get("matched_ratio")
    if matched is not None and float(matched) < TIMING_MATCHED:
        out.append("timing")
    cer = clip.get("cer_excess")
    if cer is not None and float(cer) > CER_EXCESS:
        out.append("content")
    cen = clip.get("centroid_ratio")
    if cen is not None and not (CENTROID_LOW <= float(cen) <= CENTROID_HIGH):
        out.append("timbre")
    return out


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """分類済みの clip を集計する。**どの clip かが分かるよう tag を残します。**"""
    counts: dict[str, int] = {}
    tags: dict[str, list[str]] = {}
    clean = 0
    for r in rows:
        cats = list(r.get("categories") or [])
        if not cats:
            clean += 1
        for c in cats:
            counts[c] = counts.get(c, 0) + 1
            tags.setdefault(c, []).append(str(r.get("tag")))
    return {"n_total": len(rows), "n_clean": clean, "counts": counts, "tags": tags}


def main() -> int:
    import argparse
    import json

    import numpy as np
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, help="測定 JSON の置き場")
    ap.add_argument("--dir", required=True, help="変換結果のディレクトリ")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    m = Path(a.metrics)

    def load(name):
        p = m / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    by_tag: dict[str, dict[str, Any]] = {}

    def put(tag, **kw):
        by_tag.setdefault(str(tag), {"tag": str(tag)}).update(kw)

    pitch = load("pitch.json")
    if pitch:
        for c in pitch["clips"]:
            put(c["tag"], median_abs_semitones=c.get("median_abs_semitones"),
                uv_agree=c.get("uv_agree"))
    timing = load("timing.json")
    if timing:
        for c in timing["clips"]:
            tag = c["file"].replace("_source.wav", "").split("__")[-1]
            put(tag, matched_ratio=c.get("matched_ratio"))
    cer = load("cer.json")
    if cer:
        for c in cer["clips"]:
            tag = str(c.get("file", "")).split("__")[-1]
            put(tag, cer_excess=c.get("cer_excess_over_ceiling"))

    # 波形そのものから silence / nonfinite / 明るさを見る。
    for conv in sorted(Path(a.dir).glob("*_converted.wav")):
        stem = conv.name[: -len("_converted.wav")]
        tag = stem.split("__")[-1]
        w, sr = sf.read(conv, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        put(tag, peak=float(np.abs(w).max()), finite=bool(np.all(np.isfinite(w))))
        ceil = conv.with_name(f"{stem}_vocoder_only.wav")
        if ceil.exists():
            from tools.audio_metrics import band_profile
            c1 = band_profile(w, sr).get("centroid_hz")
            g, gsr = sf.read(ceil, dtype="float32", always_2d=False)
            if g.ndim > 1:
                g = g.mean(axis=1)
            c2 = band_profile(g, gsr).get("centroid_hz")
            if c1 and c2:
                put(tag, centroid_ratio=float(c1) / float(c2))

    rows = []
    for tag in sorted(by_tag):
        clip = by_tag[tag]
        clip["categories"] = classify(clip)
        rows.append(clip)

    s = summarise(rows)
    print("=== failure taxonomy ===")
    print(f"  {s['n_total']} clip 中 {s['n_clean']} 本は分類なし")
    for cat in sorted(s["counts"], key=lambda k: -s["counts"][k]):
        print(f"  {cat:10s} {s['counts'][cat]:3d} 本  {s['tags'][cat][:6]}")
    if not s["counts"]:
        print("  （機械的に判定できる破綻はありませんでした。**聴感は blind test の担当**）")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": s, "clips": rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
