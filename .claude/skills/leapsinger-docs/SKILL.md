---
name: leapsinger-docs
description: doc/ 配下と README / CLAUDE.md を更新するときの作法。確度ラベル（確認済み・決定・推奨・見積もり・仮説・未実装・要ユーザー判断）の使い分け、主張してよい範囲、複数文書の同期、リンク検証。実験結果を書き足すとき、実装状況を更新するとき、マイルストーンの完了を宣言するときに使う。
---

# ドキュメント更新の作法

`doc/` は「何が事実で、何がまだ確かめられていないか」を読者が区別できることに価値がある。
この境界を曖昧にする更新は、内容が正しくても劣化とみなす。

## 1. 確度ラベルを必ず付ける

| ラベル | 使ってよい条件 |
|---|---|
| **確認済み** | ローカルのコード・実行 artifact・一次資料の**いずれかを示せる**とき |
| **決定** | この開発で採用する方針として確定したとき |
| **推奨** | 現時点の技術判断。実験で覆り得るもの |
| **見積もり** | 実測前のデータ量・GPU・時間 |
| **仮説** | 評価実験で検証すべき品質上の予想 |
| **未実装** | 設計はあるがコードまたはデータが無い |
| **要ユーザー判断** | 権利・目標品質・遅延など、利用者の決定が要る |

**迷ったら弱いほうを選ぶ。** 「確認済み」を付けるときは、根拠（コマンド、出力、ファイルパス、
コミット）を同じ段落に書く。

## 2. 主張の範囲（[`doc/svc-prior-art-license.md`](../../../doc/svc-prior-art-license.md) 6 節）

- 「Seed-VC より良い」— 同一 test set の blind comparison の後だけ。
- 「リアルタイム」— 対象ハードでの end-to-end 遅延実測と連続動作の後だけ。
- 「世界初」「唯一」— 使わない（rectified-flow SVC も harmonic modelling も先行研究がある）。
- 「1-step」— acoustic flow の step 数。pipeline 全体の話ではない。
- **合成データで通した疎通は「実データレベル」の根拠にならない。** 完了レベルを上げない。

## 3. 更新したら同期する

事実が動いたら、以下は**同時に**直す。片方だけ直すと矛盾が残る。

| 何が動いたか | 直す場所 |
|---|---|
| 実装・検証状況 | [`svc-implementation-status.md`](../../../doc/svc-implementation-status.md) の検証済み/制限付き/未検証 |
| 到達段階 | [`svc.md`](../../../doc/svc.md) の完了レベル 1〜5 |
| マイルストーン完了 | [`svc-plan.md`](../../../doc/svc-plan.md) の判定材料と現在地 |
| 見積もりの実測化 | [`svc-data-compute.md`](../../../doc/svc-data-compute.md) |
| 方針の確定 | [`svc.md`](../../../doc/svc.md) の主要な決定事項 / 未決事項 |
| 参照した根拠 | [`svc-sources.md`](../../../doc/svc-sources.md) のローカル根拠 |
| 開発上の落とし穴 | [`CLAUDE.md`](../../../CLAUDE.md) の既知の落とし穴 |
| 作業手順で分かったこと | 該当する skill（[verify](../leapsinger-verify/SKILL.md) / [experiment](../leapsinger-experiment/SKILL.md) / [tdd](../leapsinger-tdd/SKILL.md) / [vast](../vast-instance/SKILL.md)）。**文書に書いただけでは次の作業で読まれない** |
| 「常に間違い」なコマンドが判明 | `tools/hooks/guard_commands.py` にルール、`tools/hooks/test_guard.py` に**止めるケースと通すケースの両方**、CLAUDE.md の hook 一覧 |
| テスト件数・ステージ数 | CLAUDE.md / status / tdd skill。**数えずに書かない** |

新しい文書を足したら [`svc.md`](../../../doc/svc.md) の文書一覧に行を足す（索引が入口なので、
載っていない文書は無いのと同じ）。

## 4. 書き方

- シェル例は **Linux 前提で bash**。学習は vast.ai の Linux で回す。
- コマンドは必ず `uv run` 前置。依存追加は `uv add`。素の `python` / `pip` / `uv pip` は書かない。
- 表は「主張 → 根拠」の順。数値には単位と測定条件を添える。
- 既存の節番号を変えない（他文書が節名で参照している）。増やすなら末尾か既存節の中に足す。

## 5. 出す前に検証する

リンク切れは事故なので機械的に確かめる。

```bash
uv run python - <<'PY'
import glob, os, re
bad = []
files = (["README.md", "README.en.md", "CLAUDE.md"] + sorted(glob.glob("doc/*.md"))
         + sorted(glob.glob(".claude/skills/*/SKILL.md")))      # skill 内のリンクも壊れる
for f in files:
    s = open(f, encoding="utf-8").read()
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", s):
        t = m.group(2)
        if t.startswith(("http", "mailto:")):
            continue
        path, _, anchor = t.partition("#")
        tgt = os.path.normpath(os.path.join(os.path.dirname(f), path)) if path else f
        if path and not os.path.exists(tgt):
            bad.append((f, t, "file")); continue
        if anchor and os.path.exists(tgt):                       # 見出しアンカーも確かめる
            heads = re.findall(r"^#{2,6}\s+(.+)$", open(tgt, encoding="utf-8").read(), re.M)
            slugs = {re.sub(r"[^\w぀-ヿ一-鿿-]", "",
                            h.lower().replace(" ", "-").replace("`", "")) for h in heads}
            if anchor not in slugs:
                bad.append((f, t, "anchor"))
print("broken:", bad or "none")
PY
```

**アンカーまで見ること。** ファイルは在るのに見出しが無いリンクを実際に作りました
（日本語見出しの slug は記号の落ち方が直感と違います）。

日本語は表記の統一を保つ（`確認済み` などのラベルは全角のまま、コード識別子は原文のまま）。

## 6. コミットメッセージ

何を変えたかではなく、**何が事実として動いたか**を書く。検証したことと、していないことを
分けて書く。「実データは 1 度も通していない」のような制限は、書いておくと後の自分が助かる。
