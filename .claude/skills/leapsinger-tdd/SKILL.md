---
name: leapsinger-tdd
description: このリポジトリで実装を書くときの TDD の当て方。前処理・モデル・loader・抽出器など production code を書く前に必ず使う。重い事前学習モデル（ContentVec / RMVPE）や GPU に依存するコードをどうテストするか、データ契約の何をテストにするか、テストの置き場と実行方法を扱う。
---

# LeapSinger での TDD

**まず `superpowers:test-driven-development` を読むこと。** Red-Green-Refactor と「先に書いて失敗させたテストが無い実装コードは破棄する」という規律はそちらが定義します。この skill は**このリポジトリ固有の当て方**だけを扱います。

## 1. テストの置き場と実行

```bash
uv run python -m unittest test_svc_model test_svc_preprocess test_svc_dataset -v   # 単体 211 件
uv run python -m unittest test_svc_model.HarmonicSVCModelTests.test_single_item_inference_contract
uv run python tools/hooks/test_guard.py                      # hook の回帰 51 件
LEAPSINGER_INTEGRATION=1 uv run python -m unittest test_svc_preprocess_integration   # 実モデル
uv run python tools/smoke/run_smoke.py                       # 全経路（実装後の確認であってテストではない）
```

| ファイル | 範囲 |
|---|---|
| `test_svc_model.py` | モデル・loader・配線・**`train.py` の純粋関数**（`_loader_kwargs` / `perf_snapshot`） |
| `test_svc_preprocess.py` | 整列 / 部分集合 / loudness / shard / 抽出 / chunk / **phrase 命名と衝突** / **分量選択**（M1・M3） |
| `test_svc_dataset.py` | 検査 / coverage / split / report（M0）/ **GTSinger の wav 選択**（M3） |
| `test_svc_preprocess_integration.py` | **実 ContentVec / RMVPE。既定 skip**、`LEAPSINGER_INTEGRATION=1` で走る |

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
   **名前の一意性も契約にする。** 入れ子の深いコーパス（GTSinger）で名前が衝突すると
   cache を**黙って上書き**し、例外にもならず phrase 数が減るだけになる（実測: 1 歌手
   1,922 ファイルが 3 名に潰れた）。**曲名を ASCII に削らない**こと（日本語題が全部同じ名前に
   なり、1,922 中 1,723 件が潰れた）。どちらも合成データでは出ない。
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
- [ ] **学習を伴う検証は vast.ai で行った**（手元では hook が止める。[vast-instance](../vast-instance/SKILL.md)）
- [ ] hook のルールを足したなら `tools/hooks/test_guard.py` に**止めるケースと通すケースの両方**を足した

## 6. 課金中に TDD を崩したくなったら

インスタンスを止めたくない、という理由で実装を先に書きたくなります。**実際に 1 度崩しました**
（`pick_wavs`）。そのときは:

- **隠さない。** どの関数を実装先行で書いたかをコミットメッセージに書く。
- **後から足すテストは「素朴な実装なら落ちる」内容にする。** 通るだけのテストは意味がない
  （`pick_wavs` なら「先頭から取る実装だと技法が偏る」ことを落とす形にした）。
- そもそも**インスタンスを立てる前に実装を済ませる**のが正しい。課金前の手元作業に
  どこまで倒せるかを、借りる前に決めておく（[vast-instance](../vast-instance/SKILL.md) 1 節）。
