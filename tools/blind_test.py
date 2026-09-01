#!/usr/bin/env python3
"""blind preference test を用意し、集計する（[実行計画](../doc/svc-plan.md) M5 ゴール 3）。

    # 1. 聴く用のファイルと採点シートを作る（どちらが A かは分からない形で並ぶ）
    uv run python tools/blind_test.py prepare --a out/m5/leapsvc --b out/m5/seedvc \
      --a-name leapsvc --b-name seedvc --out out/m5/blind --seed 0

    # 2. out/m5/blind/sheet.csv の vote 列に A / B / tie を書いてから
    uv run python tools/blind_test.py tally --sheet out/m5/blind/sheet.csv \
      --key out/m5/blind/key.json --out out/m5/blind/result.json

**評価者は 1 名（開発者）です**（2026-09-01 決定）。**N=1 の非公式 preference test** として
報告し、**MOS とは呼びません**。それでも blind の条件は満たします。

| 条件 | 実装 |
|---|---|
| ラベルを隠す | 提示ファイル名は `pair03_A.wav` の形。**system 名が出ない** |
| 左右を入れ替える | clip ごとに A / B の割り当てを randomize。常に片方が A だと順序の癖が preference に化ける |
| 順番を混ぜる | clip の並びも shuffle。曲順で聴くと後半に慣れが出る |
| 復元できる | 割り当ては `key.json` に残す。**採点シートには入れない** |

**引き分けと未記入を捨てません。** 捨てると「差が無かった」ことが見えなくなります。

**符号検定の p 値は参考値**です。評価者 1 名なので、独立なのは clip であって人ではありません。
**「preferred」と書くときは必ず N=1 と clip 数を併記します。**
"""
from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import Any


def assign_sides(clips: Sequence[str], *, systems: tuple[str, str],
                 seed: int) -> list[dict[str, str]]:
    """clip ごとに A / B の割り当てと提示順を決める。

    **clip の並びも混ぜます。** 曲順で聴くと後半に慣れが出るためです。
    """
    if len(systems) != 2 or systems[0] == systems[1]:
        raise ValueError(f"systems は異なる 2 つにしてください（{systems} が来ました）")
    rng = random.Random(seed)
    order = list(clips)
    rng.shuffle(order)
    rows = []
    for clip in order:
        first, second = (systems if rng.random() < 0.5 else (systems[1], systems[0]))
        rows.append({"clip": str(clip), "A": first, "B": second})
    return rows


def _p_sign_test(wins: int, n: int) -> float:
    """符号検定の両側 p 値（帰無仮説: 五分五分）。"""
    if n <= 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def tally(sheet: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """採点シートを system ごとの勝ち数へ直す。

    **side ではなく system で数えます**（A が常に同じ system とは限らないため）。
    **引き分けと未記入も数えます。**
    """
    wins: dict[str, int] = {}
    ties = missing = 0
    for row in sheet:
        vote = str(row.get("vote", "")).strip()
        if not vote:
            missing += 1
            continue
        if vote.lower() == "tie":
            ties += 1
            continue
        if vote not in ("A", "B"):
            raise ValueError(
                f"vote は A / B / tie / 空 のいずれかです（{vote!r} が来ました）。"
                "system 名を直接書くと、どちら側で聴いたのかが失われます")
        winner = str(row[vote])
        wins[winner] = wins.get(winner, 0) + 1

    decisive = sum(wins.values())
    names = sorted({str(r[k]) for r in sheet for k in ("A", "B") if k in r})
    for n in names:
        wins.setdefault(n, 0)
    top = max(wins.values()) if wins else 0
    return {
        "wins": wins, "ties": ties, "n_missing": missing,
        "n_voted": decisive + ties, "n_decisive": decisive,
        "p_two_sided": _p_sign_test(top, decisive),
        "note": ("符号検定は clip を独立とみなした参考値。評価者は 1 名なので、"
                 "報告では N=1 と clip 数を必ず併記する"),
    }


def _cmd_prepare(a) -> int:
    import json
    import shutil

    a_dir, b_dir = Path(a.a), Path(a.b)
    a_clips = {p.name.split("__")[-1].replace("_converted.wav", ""): p
               for p in sorted(a_dir.glob("*_converted.wav"))}
    b_clips = {}
    for p in sorted(b_dir.rglob("*.wav")):
        tag = p.parent.name if p.parent != b_dir else p.stem
        b_clips.setdefault(tag, p)
    shared = sorted(set(a_clips) & set(b_clips))
    if not shared:
        sys.exit(f"共通の clip がありません（{a.a}: {len(a_clips)} / {a.b}: {len(b_clips)}）")
    print(f"[blind] 共通 clip {len(shared)} 本（{a.a_name} {len(a_clips)} / "
          f"{a.b_name} {len(b_clips)}）")

    rows = assign_sides(shared, systems=(a.a_name, a.b_name), seed=a.seed)
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    src = {a.a_name: a_clips, a.b_name: b_clips}
    key, sheet = [], ["pair,clip,vote  # vote に A / B / tie を書く"]
    for i, row in enumerate(rows):
        pair = f"pair{i:02d}"
        for side in ("A", "B"):
            shutil.copyfile(src[row[side]][row["clip"]], out / "audio" / f"{pair}_{side}.wav")
        key.append({"pair": pair, **row})
        sheet.append(f"{pair},{row['clip']},")
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    (out / "sheet.csv").write_text("\n".join(sheet) + "\n", encoding="utf-8")
    print(f"  -> {out}/audio （**system 名は出ません**）")
    print(f"  -> {out}/sheet.csv （vote 列を埋める）")
    print(f"  -> {out}/key.json （集計まで見ないこと）")
    return 0


def _cmd_tally(a) -> int:
    import csv
    import json

    key = {r["pair"]: r for r in json.loads(Path(a.key).read_text(encoding="utf-8"))}
    sheet = []
    with open(a.sheet, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pair = (row.get("pair") or "").strip()
            if not pair or pair not in key:
                continue
            sheet.append({"clip": key[pair]["clip"], "A": key[pair]["A"], "B": key[pair]["B"],
                          "vote": (row.get("vote") or "").strip()})
    if not sheet:
        sys.exit("採点シートに有効な行がありません")

    rep = tally(sheet)
    rep["rows"] = sheet
    print("=== blind preference（N=1、非公式）===")
    for name, w in sorted(rep["wins"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:10s} {w:3d} 勝")
    print(f"  引き分け {rep['ties']} / 未記入 {rep['n_missing']} / 判定 {rep['n_decisive']}")
    print(f"  符号検定 p = {rep['p_two_sided']:.4f}（参考値）")
    print(f"\n  ** {rep['note']} **")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prepare", help="聴く用のファイルと採点シートを作る")
    p1.add_argument("--a", required=True, help="片方の変換結果ディレクトリ")
    p1.add_argument("--b", required=True, help="もう片方")
    p1.add_argument("--a-name", default="A系")
    p1.add_argument("--b-name", default="B系")
    p1.add_argument("--out", required=True)
    p1.add_argument("--seed", type=int, default=0)
    p1.set_defaults(func=_cmd_prepare)

    p2 = sub.add_parser("tally", help="採点シートを集計する")
    p2.add_argument("--sheet", required=True)
    p2.add_argument("--key", required=True)
    p2.add_argument("--out", default=None)
    p2.set_defaults(func=_cmd_tally)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
