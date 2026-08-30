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
    else:
        reasons.append("empty=no finite samples")

    return reasons
