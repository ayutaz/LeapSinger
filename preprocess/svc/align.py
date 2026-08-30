"""SSL の frame grid を mel の frame grid へ合わせる。

SSL は 16 kHz・stride 320 = 50 Hz、mel grid は 44,100 / 256 = 172.265625 Hz で、
比 3.4453125 は整数になりません。方式は `doc/svc-content-encoder.md` 6 節で
**left（直前保持・左寄せ繰り返し）**に決めています。実測した性質:

    方式      left 基準の先読み   ブレンドされるフレーム
    left            0 ms                0%
    nearest        20 ms                0%
    linear         20 ms             99.4%

left だけが「先読み 0」かつ「元の SSL ベクトルをそのまま保持」を同時に満たします。
"""
from __future__ import annotations

import numpy as np


def align_left(x: np.ndarray, target_len: int) -> np.ndarray:
    """`[T_src, C]` を `[target_len, C]` へ直前保持で整列する。

    出力 `t` は入力 `floor(t * T_src / target_len)` をそのまま複製したものです
    （so-vits-svc の `repeat_expand_2d_left` と同じ index 演算）。値を混ぜないので
    出力は必ず入力に実在するベクトルで、未来の frame も参照しません。

    index は整数演算で求めます。浮動小数の丸めが混ざると同じ入力でも環境差で
    結果が変わり得るためです（データ契約は再実行の bit 一致を要求します）。
    """
    if x.ndim != 2:
        raise ValueError(f"content must be [T, C]; got shape {x.shape}")
    src_len = int(x.shape[0])
    if src_len == 0:
        raise ValueError("content must have at least one frame")
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be positive; got {target_len}")

    idx = np.minimum(np.arange(target_len) * src_len // target_len, src_len - 1)
    return x[idx]
