---
name: leapsinger-tdd
description: このリポジトリで実装を書くときの TDD の当て方。前処理・モデル・loader・抽出器など production code を書く前に必ず使う。重い事前学習モデル（ContentVec / RMVPE）や GPU に依存するコードをどうテストするか、データ契約の何をテストにするか、テストの置き場と実行方法を扱う。
---

# LeapSinger での TDD

**まず `superpowers:test-driven-development` を読むこと。** Red-Green-Refactor と「先に書いて失敗させたテストが無い実装コードは破棄する」という規律はそちらが定義します。この skill は**このリポジトリ固有の当て方**だけを扱います。

## 1. テストの置き場と実行

```bash
uv run python -m unittest test_svc_model -v                  # SVC 関連
uv run python -m unittest test_svc_model.HarmonicSVCModelTests.test_single_item_inference_contract
uv run python tools/hooks/test_guard.py                      # hook の回帰
uv run python tools/smoke/run_smoke.py                       # 全経路（実装後の確認であってテストではない）
```

- top-level の `test_<領域>.py` に置く。`unittest discover` は top-level の `test_*.py` しか拾わないので、
  新しい領域を足したら**その名前を [CLAUDE.md](../../../CLAUDE.md) のテスト節に追記する**（さもないと誰も走らせない）。
- `unittest discover` は hook で止めています。**「discover が通った」を検証の根拠にしない。**

## 2. 重いモデルに依存するコードのテスト

M1 の抽出器は ContentVec（数百 MB）と RMVPE（181 MB）に依存します。**これらを落とさずに走るテストを既定にする。**

| 層 | 何をテストするか | 依存 |
|---|---|---|
| 単体（既定） | 整形・整列・正規化・命名・manifest・エラー処理 | **なし**。encoder は差し替え可能な口から fake を注入する |
| 統合（明示指定時のみ） | 実 encoder の出力 shape・dtype・値域 | 実モデル |

**設計への含意:** encoder を関数の内側で `from_pretrained` すると単体テストが書けません。**テストが書きにくいと感じたら設計が悪い**ので、encoder を引数で受ける形に直す（`superpowers` の "Must mock everything → Code too coupled. Use dependency injection."）。

fake encoder は「決められた frame 数の決められた次元を返すだけ」で十分です。実モデルの数値を模倣する必要はありません。**整列と契約をテストしているのであって、encoder の中身をテストしているのではない。**

GPU も同じです。単体テストは CPU で通ること。`torch.cuda.is_available()` に依存する分岐を書いたら、その分岐自体をテスト可能にする。

## 3. このリポジトリで最初にテストにすべき不変条件

データ契約は「黙って直さない」ことに価値があります。**その拒否をテストにする。**

1. **フレーム完全一致** — `content` `f0_interp` `uv` `loudness` `mel` の `T` が 1 でもずれたら例外。
   ずれを許容するコードを書いていないことを、ずれた入力を渡して確認する。
2. **補間の境界** — SSL 50 Hz と mel 172.265625 Hz は整数比になりません。
   端数が出る長さ（例: 短い phrase）で `T` がちょうど mel と一致することをテストする。
3. **phrase 命名** — `{song}_{NNNN}`。崩れると `_song_of()` の曲単位分割が効かず leakage する。
4. **決定性** — 同じ入力・同じ設定で 2 回実行して **bit 一致**。ランダム性が混ざる箇所（256 次元の
   部分集合など）は seed から決まることをテストする。
5. **manifest の完全性** — 再現に要る項目が欠けていないこと。項目を増やしたらテストも増やす。
6. **既存 SVS 経路を壊していないこと** — SVC の変更で `dataset.py` / `preprocess` / export の
   契約が変わっていないこと。

## 4. やりがちな失敗

| 失敗 | なぜ駄目か |
|---|---|
| 実音声ファイルを fixture にコミットする | `.gitignore` が `*.wav` を除外している。権利の問題もある。**合成波形を生成する**（`tools/smoke/gen_synth_data.py` が例） |
| テストで 181 MB を落とす | 単体テストが遅く、ネットワークに依存する。統合テスト側へ隔離する |
| `run_smoke.py` が通ったことをテストの代わりにする | あれは配線の疎通確認であって、契約のテストではない。**両方要る** |
| 合成音声で通ったことを品質の根拠にする | [leapsinger-docs](../leapsinger-docs/SKILL.md) の主張規則違反。完了レベルは上がらない |
| 数値の期待値をハードコードして後で書き換える | テストが実装に追従したら意味がない。契約（shape・不変条件・例外）でテストする |

## 5. 完了の判定

`superpowers:test-driven-development` の Verification Checklist に加えて、このリポジトリでは:

- [ ] 単体テストが**ネットワークと GPU 無しで**通る
- [ ] 新しい `test_*.py` を [CLAUDE.md](../../../CLAUDE.md) のテスト節に追記した
- [ ] `uv run python tools/smoke/run_smoke.py` が通る（既存経路を壊していない）
- [ ] 実装で分かった落とし穴を CLAUDE.md の「既知の落とし穴」へ書いた
