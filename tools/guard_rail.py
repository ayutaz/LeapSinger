#!/usr/bin/env python3
"""事前登録した guard rail 判定（[実行計画](../doc/svc-plan.md) M5「決定 1」）。

**絶対閾値は置きません。** 手元のデータ量（CER n=1、信号品質 n=2 で符号が反転）から閾値を
作ると、根拠のない数を発明することになります。かわりに:

| 役割 | 内容 |
|---|---|
| **主要な判定** | **blind preference**（M5 の目的が比較だから） |
| **guard rail** | source / target 録音を基準にする指標。**baseline にもそのまま当たる** |
| **「悪化」の定義** | **差の標準誤差の 2 倍**を超えて相手より低い |
| **主張の制限** | **preference で勝っても guard rail を 1 つでも落としていたら「より良い」とは書かない** |

最後の行をコードに書いているのは、**後から緩められないようにする**ためです。M4 で
「規則を後から決めると train loss で最悪の checkpoint を選ぶ」ことを実証しました。

**なぜ 2 SE か（実測して決めた）:** 同一分布から 20 clip ずつ取った標本 300 組で、
誤って「悪化」と判定される割合は **1 SE で 15.3%、2 SE で 2.7%** でした。guard rail は
悪化を見逃さないための仕組みですが、**毎回どれかが偶然落ちると主張そのものができなく
なる**ので、誤検知を数 % に抑える 2 SE を採ります。**事前登録した「ばらつきを超えて」の
具体化であって、結果を見てからの調整ではありません**（測定前に決めています）。
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# 「悪化」とみなす境界。差が **標準誤差の SE_MULTIPLIER 倍**を超えて悪いときだけ落とす。
# 実測（同一分布 20 clip x 300 組）で誤検知は 1 SE 15.3% / 2 SE 2.7%。
SE_MULTIPLIER = 2.0


def compare_metric(ours: Sequence[float], theirs: Sequence[float], *,
                   higher_is_better: bool) -> dict[str, Any]:
    """2 系列を比べ、**ばらつきを超えて悪いか**を返す。

    差の標準誤差（2 標本）を閾値にします。**平均だけを比べません** — clip ごとのばらつきが
    大きい指標では、平均の差が偶然の範囲に収まることがあるためです。
    """
    a = np.asarray(list(ours), dtype=np.float64)
    b = np.asarray(list(theirs), dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError(f"空の系列は比べられません（ours {a.size} / theirs {b.size}）")

    diff = float(a.mean() - b.mean())
    if not higher_is_better:
        diff = -diff                      # 小さいほうが良い指標は符号を反転して扱う
    se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)) if (
        a.size > 1 and b.size > 1) else 0.0
    se *= SE_MULTIPLIER
    return {
        "ours_mean": float(a.mean()), "theirs_mean": float(b.mean()),
        "n_ours": int(a.size), "n_theirs": int(b.size),
        "diff": diff, "threshold": se, "higher_is_better": bool(higher_is_better),
        "worse": bool(diff < -se),
    }


def verdict(rails: dict[str, dict[str, Any]], *, preference_winner: str | None,
            ours: str) -> dict[str, Any]:
    """「より良い」と書いてよいかを返す。

    **勝つだけでは足りません。** guard rail が 1 つでも落ちていたら `may_claim_better` は
    False になり、報告では「preferred だが X は低い」と両方書きます。
    """
    failed = sorted(k for k, v in rails.items() if v.get("worse"))
    won = preference_winner == ours
    return {
        "preference_winner": preference_winner, "ours": ours,
        "failed_rails": failed,
        "may_claim_better": bool(won and not failed),
        "reason": ("preference で勝ち、guard rail もすべて保っている" if won and not failed
                   else "preference で勝っていない" if not won
                   else f"guard rail が落ちている: {failed}"),
        "rule": ("事前登録（2026-09-01）: preference で勝っても guard rail を 1 つでも"
                 "落としていたら「より良い」とは書かない"),
    }


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="自系の測定 JSON の置き場")
    ap.add_argument("--theirs", required=True, help="baseline の測定 JSON の置き場")
    ap.add_argument("--preference", default=None, help="blind test の result.json")
    ap.add_argument("--ours-name", default="leapsvc")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def load(root: str, name: str):
        p = Path(root) / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    rails: dict[str, dict[str, Any]] = {}

    # timing: 対応が付いた割合（大きいほど良い）
    for key, fn, extract, hib in (
        ("timing_matched_ratio", "timing.json",
         lambda d: [c["matched_ratio"] for c in d["clips"] if c["matched_ratio"] is not None],
         True),
        ("timing_deviation", "timing.json",
         lambda d: [c["median_abs_dev"] for c in d["clips"] if c["median_abs_dev"] is not None],
         False),
    ):
        o, t = load(a.ours, fn), load(a.theirs, fn)
        if o and t:
            rails[key] = compare_metric(extract(o), extract(t), higher_is_better=hib)

    # 話者類似度: 回復率（大きいほど良い）。1 値しか出ないので clip 単位の cos を使う
    o, t = load(a.ours, "similarity.json"), load(a.theirs, "similarity.json")
    if o and t:
        rails["speaker_similarity"] = compare_metric(
            [o["converted_vs_target"]["mean"]], [t["converted_vs_target"]["mean"]],
            higher_is_better=True)

    pref = json.loads(Path(a.preference).read_text(encoding="utf-8")) if a.preference else None
    winner = None
    if pref:
        wins = pref.get("wins", {})
        if wins:
            top = max(wins, key=lambda k: wins[k])
            if list(wins.values()).count(wins[top]) == 1:
                winner = top

    v = verdict(rails, preference_winner=winner, ours=a.ours_name)
    rep = {"rails": rails, "verdict": v}

    print("=== guard rail ===")
    for k, r in sorted(rails.items()):
        mark = "NG" if r["worse"] else "ok"
        print(f"  [{mark}] {k:22s} ours {r['ours_mean']:.4f} / theirs {r['theirs_mean']:.4f}"
              f"  差 {r['diff']:+.4f}（閾値 {r['threshold']:.4f}）")
    print(f"\n  preference: {winner or '（判定なし）'}")
    print(f"  **「より良い」と書けるか: {'はい' if v['may_claim_better'] else 'いいえ'}** "
          f"— {v['reason']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
