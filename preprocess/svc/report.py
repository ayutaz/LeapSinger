"""M0 の成果物を 1 度に作る（[実行計画](../../doc/svc-plan.md) M0）。

計画の成果物は「dataset ledger、split list、reject list、coverage 集計」です。道具が関数として
在るだけでは素材が届くたびにつなぎを書くことになるので、検査から split までを 1 本にまとめます。

音声の読み込みは呼び出し側から渡します（`load_audio(name) -> (wav, sr)`）。ファイル I/O を
中に抱えないので、偽のローダーで契約をテストできます。

**未実装:** 音域・技法の coverage 集計はここに含めていません。音域は F0 が要り、F0 抽出には
RMVPE（181 MB）が要るためです。技法はラベルのある corpus（GTSinger / VocalSet）でのみ集計できます。
どちらも `preprocess/svc/coverage.py` の関数を別途呼びます。
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Mapping

import numpy as np

from .audit import AuditThresholds, audit_clip
from .coverage import label_seconds
from .split import split_by_group

LoadAudio = Callable[[str], "tuple[np.ndarray, int]"]


def build_report(names: Mapping[str, str], load_audio: LoadAudio, *,
                 expected_sr: int, seed: int, eval_groups: int, test_groups: int,
                 thresholds: AuditThresholds | None = None,
                 labels: Mapping[str, str] | None = None) -> dict:
    """全クリップを検査し、通ったものだけで split を作る。

    `names` は `{phrase 名: group 名}`。group は曲か収録セッションです。
    `labels` を渡すと、受理したクリップだけでラベルごとの滞在秒数も出します
    （技法・性別など。M0 ゴール 3 の「発声スタイルの coverage」）。

    弾いた素材は split に入れません。残すと学習が起動時に落ちます。全クリップが弾かれた場合は
    例外にします。空の split を黙って返すと、検査が厳しすぎるのか素材が壊れているのかが
    分からないまま先へ進んでしまうためです。
    """
    th = thresholds or AuditThresholds()
    rejects: dict[str, list[str]] = {}
    accepted: dict[str, str] = {}
    accepted_dur: dict[str, float] = {}
    accepted_sec = rejected_sec = 0.0

    for name in sorted(names):
        wav, sr = load_audio(name)
        duration = len(wav) / float(sr) if sr else 0.0
        reasons = audit_clip(wav, sr, expected_sr=expected_sr, thresholds=th)
        if reasons:
            rejects[name] = reasons
            rejected_sec += duration
        else:
            accepted[name] = names[name]
            accepted_dur[name] = duration
            accepted_sec += duration

    if not accepted:
        raise ValueError(
            f"全 {len(names)} クリップが除外されました。検査が厳しすぎるか素材が壊れています。"
            f"理由の例: {next(iter(rejects.values()), [])}")

    split = split_by_group(accepted, seed=seed, eval_groups=eval_groups,
                           test_groups=test_groups)

    coverage: dict[str, dict[str, float]] = {}
    if labels is not None:
        # 弾いた素材は数えない。混ぜると、実際に学習する分布と食い違う。
        pairs = [(labels[n], accepted_dur[n]) for n in sorted(accepted) if n in labels]
        coverage["labels"] = label_seconds([l for l, _ in pairs], [d for _, d in pairs])

    return {
        "rejects": rejects,
        "coverage": coverage,
        "split": split,
        "totals": {
            "accepted": len(accepted), "rejected": len(rejects),
            "accepted_sec": accepted_sec, "rejected_sec": rejected_sec,
            "groups": len(set(accepted.values())),
        },
        "manifest": {
            "seed": seed, "eval_groups": eval_groups, "test_groups": test_groups,
            "expected_sr": expected_sr, "thresholds": asdict(th),
        },
    }
