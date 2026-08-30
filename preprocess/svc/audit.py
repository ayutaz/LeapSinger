"""素材 1 クリップの品質検査（[実行計画](../../doc/svc-plan.md) M0 ゴール 2）。

合格なら空リスト、駄目なら**除外理由の一覧**を返します。1 つ直したら次が出る、では検査が
何往復もするので、当てはまる理由はまとめて返します。理由には実測値を添えます
（閾値を調整するには「clipping」だけでは足りないため）。

戻り値の文字列がそのまま reject list の行になります。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AuditThresholds:
    """既定値は「明らかにおかしいものだけを弾く」水準。素材に合わせて締めてください。"""

    clip_level: float = 0.999          # これ以上を full-scale とみなす
    max_clipping_ratio: float = 1e-4   # full-scale が全体の 0.01% を超えたら潰れている
    silence_level: float = 1e-5        # これ未満を無音とみなす（mel の floor と同じ）
    max_silence_ratio: float = 0.5     # 半分以上が無音なら素材として使えない
    max_dc_offset: float = 0.02        # 直流オフセット
    min_sec: float = 0.3               # configs の data.min_sec と揃えた
    min_bandwidth_hz: float | None = None   # None で無効。実音声には mel.fmax と同じ 16000 を推奨


def audit_clip(wav: np.ndarray, sr: int, *, expected_sr: int,
               thresholds: AuditThresholds | None = None) -> list[str]:
    """1 クリップを検査して除外理由を返す。合格なら空リスト。"""
    th = thresholds or AuditThresholds()
    wav = np.asarray(wav)
    if wav.ndim != 1:
        raise ValueError(f"wav must be mono 1-D; got shape {wav.shape}")

    reasons: list[str] = []
    if int(sr) != int(expected_sr):
        # mel 設定は前処理・loader・励起で共有される。sr の取り違えは静かに全部を壊す。
        reasons.append(f"sample_rate={int(sr)} expected={int(expected_sr)}")

    finite = np.isfinite(wav)
    if not finite.all():
        reasons.append(f"non_finite={int((~finite).sum())}")
    w = wav[finite].astype(np.float64)

    duration = wav.size / float(sr) if sr else 0.0
    if duration < th.min_sec:
        reasons.append(f"too_short={duration:.3f}s min={th.min_sec}s")

    if w.size:
        clipped = float((np.abs(w) >= th.clip_level).mean())
        if clipped > th.max_clipping_ratio:
            reasons.append(f"clipping={clipped:.4%} max={th.max_clipping_ratio:.4%}")

        silent = float((np.abs(w) < th.silence_level).mean())
        if silent > th.max_silence_ratio:
            reasons.append(f"silence={silent:.1%} max={th.max_silence_ratio:.1%}")

        dc = abs(float(w.mean()))
        if dc > th.max_dc_offset:
            reasons.append(f"dc_offset={dc:.5f} max={th.max_dc_offset}")

        if th.min_bandwidth_hz is not None:
            # 既定は無効。純音のような合成信号は高域が無くて当然で、常時有効だと誤検出になる。
            bandwidth = effective_bandwidth_hz(w, sr)
            if bandwidth < th.min_bandwidth_hz:
                reasons.append(f"band_limited={bandwidth:.0f}Hz "
                               f"min={th.min_bandwidth_hz:.0f}Hz")
    else:
        reasons.append("empty=no finite samples")

    return reasons


def effective_bandwidth_hz(wav: np.ndarray, sr: int, *, floor_db: float = -60.0) -> float:
    """実際にエネルギーが入っている上限周波数を返す。

    低い sample rate から上げただけの素材を見抜くために使います。`mel.fmax` は 16,000 Hz なので、
    24 kHz 音源（Nyquist 12 kHz）を 44.1 kHz へ上げても 12〜16 kHz は空のままです。それを混ぜて
    学習すると、その帯域を「無い」と学習してこもった出力になります。

    スペクトルをピークからの相対 dB で見て、`floor_db` を上回る最も高い帯域の周波数を返します。
    ビンをまとめて平均するのは、白色雑音のようにビン単位では暴れる信号でも安定させるためです。
    """
    w = np.asarray(wav, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size < 2:
        return 0.0
    spec = np.abs(np.fft.rfft(w * np.hanning(w.size)))
    freqs = np.fft.rfftfreq(w.size, 1.0 / float(sr))

    group = max(1, spec.size // 512)                  # 512 帯域くらいに束ねる
    usable = (spec.size // group) * group
    band = spec[:usable].reshape(-1, group).mean(axis=1)
    band_hz = freqs[:usable].reshape(-1, group).mean(axis=1)

    peak = float(band.max())
    if peak <= 0.0:
        return 0.0
    db = 20.0 * np.log10(np.maximum(band, 1e-20) / peak)
    above = np.flatnonzero(db > floor_db)
    return float(band_hz[above[-1]]) if above.size else 0.0
