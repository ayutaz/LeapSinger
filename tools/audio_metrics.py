"""音の明るさを数値にする。検証で「こもり」を検知するための指標。

M3 の検証は content cos / F0 相関 / V/UV だけで、**高域の欠落を検知できませんでした**。
推論側の loudness 条件がずれて spectral centroid が 620 → 368 Hz に落ちたとき、
content cos は 0.8217 → 0.8096 としか動きません（実測）。耳では明らかに違う音です。

内容・音高・有声区間が保たれていても音が鈍ることはあるので、**別の軸として**測ります。
"""
from __future__ import annotations

import numpy as np

# NHVSing V3 の mel は 40-16000 Hz なので、それより上は元から入っていない。
BANDS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000))


def band_profile(wav: np.ndarray, sr: int, *, n_fft: int = 4096) -> dict:
    """有声側フレームの帯域エネルギー比と spectral centroid。

    無音や息継ぎで薄まらないよう、**エネルギーが中央値以上のフレームだけ**を平均します。
    窓に満たない長さでは `{}` を返します（判断材料にならないため、0 で埋めない）。
    """
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    win = np.hanning(n_fft)
    frames = [wav[i:i + n_fft] * win for i in range(0, max(0, len(wav) - n_fft), n_fft // 2)]
    if not frames:
        return {}
    mags = np.abs(np.fft.rfft(np.array(frames), axis=1))
    energy = mags.sum(axis=1)
    m = mags[energy >= np.median(energy)].mean(axis=0) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    power = m ** 2
    total = float(power.sum())
    return {
        "bands": {f"{lo // 1000}-{hi // 1000}k":
                  float(power[(freqs >= lo) & (freqs < hi)].sum() / total) for lo, hi in BANDS},
        "centroid_hz": float((freqs * power).sum() / total),
    }


def format_profiles(profiles: dict[str, dict]) -> str:
    """`{名前: band_profile(...)}` を 1 つの表にする。"""
    named = {k: v for k, v in profiles.items() if v}
    if not named:
        return "(帯域を測れる長さがありません)"
    keys = list(next(iter(named.values()))["bands"])
    head = f"{'':16}" + "".join(f"{k:>8}" for k in keys) + f"{'centroid':>11}"
    rows = [f"{name:<16}" + "".join(f"{v * 100:7.1f}%" for v in p["bands"].values())
            + f"{p['centroid_hz']:9.0f}Hz" for name, p in named.items()]
    return "\n".join([head, *rows])
