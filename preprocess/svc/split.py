"""train / eval / test を **group 単位**で分ける（[実行計画](../../doc/svc-plan.md) M0 ゴール 4）。

group は曲、または収録セッションです。フレーズ単位で切ると同じ曲が train と test の両方に
入り、leakage で性能を過大評価します。`dataset.py` の `_song_of()` が phrase 名
`{song}_{NNNN}` から曲名を取り出すので、通常はそれを group に使います。収録セッションで
分けたい場合は、名前から導けないので呼び出し側が group を渡します。

乱数は `random.Random`（Mersenne Twister）を使います。numpy の `Generator` と違って
バージョン間でストリームが保証されるため、seed だけで split を再現できます。
`dataset.py` の hold-out 選択と同じ流儀です。
"""
from __future__ import annotations

import random
from typing import Mapping


def split_by_group(names: Mapping[str, str], *, seed: int,
                   eval_groups: int, test_groups: int) -> dict[str, list[str]]:
    """`{phrase 名: group 名}` を train / eval / test に分ける。

    同じ group が 2 つの split に跨がることはありません。各 split 内の名前は昇順です
    （split list はファイルに書いて差分を読むものなので、順序が安定しないと差分が意味を失う）。
    """
    if not names:
        raise ValueError("split_by_group needs at least one phrase")
    eval_groups, test_groups = int(eval_groups), int(test_groups)
    if eval_groups < 0 or test_groups < 0:
        raise ValueError("eval_groups and test_groups must be non-negative")

    groups = sorted(set(names.values()))
    if eval_groups + test_groups >= len(groups):
        raise ValueError(
            f"train に group が残りません: {len(groups)} groups では "
            f"eval {eval_groups} + test {test_groups} を取れません")

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    held_test = set(shuffled[:test_groups])
    held_eval = set(shuffled[test_groups:test_groups + eval_groups])

    out: dict[str, list[str]] = {"train": [], "eval": [], "test": []}
    for name in sorted(names):
        group = names[name]
        key = "test" if group in held_test else "eval" if group in held_eval else "train"
        out[key].append(name)
    return out
