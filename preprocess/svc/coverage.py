"""音域の coverage 集計（[実行計画](../../doc/svc-plan.md) M0 ゴール 3）。

高音や裏声が薄い素材で学習すると、そこだけ崩れます。学習を始める前に分布を見て、
足りない音域を素材の追加や取捨で埋めるための集計です。

**未実装:** 発声スタイル（chest / falsetto / breathy / 強声 / 弱声）の coverage は、
音声から自動で測る手段がありません。ラベルを付けるか分類器を用意する必要があり、
現状は手作業の注釈を前提にします。ここで出せるのは音域と有声時間だけです。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _voiced(f0_hz: np.ndarray, uv: np.ndarray) -> np.ndarray:
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    if f0_hz.shape != uv.shape:
        raise ValueError(f"f0 and uv must have the same shape; got {f0_hz.shape} and {uv.shape}")
    return f0_hz[(uv > 0.5) & np.isfinite(f0_hz) & (f0_hz > 0)]


def pitch_band_seconds(f0_hz: np.ndarray, uv: np.ndarray, *, frame_rate: float,
                       edges_hz: Sequence[float]) -> dict[str, float]:
    """有声フレームを `edges_hz` で区切った帯域ごとに数え、**滞在秒数**を返す。

    境界値はその**上側**の帯域に入れます（`edges_hz=[200]` なら 200.0 Hz は `200-` 側）。
    無声フレームは数えません。音域を測るのに無声区間を混ぜても意味がないためです。
    """
    edges = [float(e) for e in edges_hz]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"edges_hz must be strictly increasing; got {edges}")

    voiced = _voiced(f0_hz, uv)
    labels = ([f"-{edges[0]:.0f}"]
              + [f"{a:.0f}-{b:.0f}" for a, b in zip(edges, edges[1:])]
              + [f"{edges[-1]:.0f}-"])
    per_frame = 1.0 / float(frame_rate)
    counts = np.searchsorted(edges, voiced, side="right") if voiced.size else np.array([], int)
    return {label: float((counts == i).sum()) * per_frame for i, label in enumerate(labels)}


def voiced_range(f0_hz: np.ndarray, uv: np.ndarray, *, frame_rate: float) -> dict[str, float]:
    """有声区間の音域を返す。

    広さは半音（`span_semitones`）でも出します。Hz の差は低音と高音で意味が変わるため、
    音域の広さは半音のほうが読めます。有声フレームが 1 つも無い場合は全て 0 を返します
    （例外にすると全曲の集計が 1 クリップで止まるため）。
    """
    voiced = _voiced(f0_hz, uv)
    if voiced.size == 0:
        return {"voiced_sec": 0.0, "p05_hz": 0.0, "p50_hz": 0.0, "p95_hz": 0.0,
                "min_hz": 0.0, "max_hz": 0.0, "span_semitones": 0.0}
    p05, p50, p95 = (float(x) for x in np.percentile(voiced, [5, 50, 95]))
    lo, hi = float(voiced.min()), float(voiced.max())
    return {
        "voiced_sec": float(voiced.size) / float(frame_rate),
        "p05_hz": p05, "p50_hz": p50, "p95_hz": p95,
        "min_hz": lo, "max_hz": hi,
        "span_semitones": 12.0 * float(np.log2(hi / lo)) if lo > 0 else 0.0,
    }


def label_seconds(labels: Sequence[str],
                  durations: Sequence[float]) -> dict[str, float]:
    """ラベルごとの滞在秒数を、**多い順**に返す。

    発声スタイル（chest / falsetto / breathy など）の coverage 用です。技法ラベルを持つ
    corpus（GTSinger の 6 種、VocalSet の 17 種）ならこれで偏りが分かります。

    区間の長さで重み付けします。区間数を数えると、短い区間が多いだけで「多い」ことに
    なってしまうためです。同数のときはラベル名の昇順にします（並びが安定しないと
    集計の差分が読めない）。
    """
    labels = list(labels)
    durations = [float(d) for d in durations]
    if len(labels) != len(durations):
        raise ValueError(f"labels and durations must match; got {len(labels)} and {len(durations)}")
    if any(d < 0 for d in durations):
        raise ValueError("durations must be non-negative")

    total: dict[str, float] = {}
    for label, duration in zip(labels, durations):
        total[str(label)] = total.get(str(label), 0.0) + duration
    return dict(sorted(total.items(), key=lambda kv: (-kv[1], kv[0])))
