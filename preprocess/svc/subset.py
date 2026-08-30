"""ContentVec 768 次元から学習に使う部分集合を選ぶ。

`doc/svc-content-encoder.md` 3 節。ランダムに選んだ部分集合を固定するだけで、生の 768 次元より
timbre 漏れが減ります（Interspeech 2025 の同一 baseline 比較で out-of-domain の SMOS
3.512 -> 4.038、内容の CMOS はほぼ不変）。学習される射影ではなく**ただの列の選択**なので、
実装コストも推論コストもゼロです。

**再現性:** manifest には seed だけでなく**選ばれた index そのもの**を記録します。numpy の
`Generator` は分布メソッドのストリームがバージョン間で変わり得るため、seed だけでは
将来同じ index を再生成できる保証がありません。index を残せば、その shard は常に再現できます。
"""
from __future__ import annotations

import numpy as np


def subset_indices(total: int, n: int, seed: int) -> np.ndarray:
    """`total` 次元から `n` 次元を選び、**昇順**の index を返す。

    昇順にするのは manifest に載せて人が読むものだからです（並びが安定しないと差分が読めない）。
    列の順序自体に意味は無いので、並べ替えても表現は変わりません。
    """
    total = int(total)
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if n > total:
        raise ValueError(f"cannot select {n} dims out of {total}")
    idx = np.random.default_rng(seed).choice(total, size=n, replace=False)
    return np.sort(idx).astype(np.int64)


def apply_subset(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """`[T, C]` から `indices` の列だけを取り出して `[T, len(indices)]` を返す。"""
    if x.ndim != 2:
        raise ValueError(f"content must be [T, C]; got shape {x.shape}")
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1-D; got shape {indices.shape}")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= x.shape[1]):
        raise ValueError(
            f"indices out of range for width {x.shape[1]}: "
            f"[{int(indices.min())}, {int(indices.max())}]")
    return x[:, indices]
