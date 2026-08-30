---
name: leapsinger-experiment
description: LeapSinger の学習実験（SVS / SVC の base 学習、fine-tune、ablation）を最初から最後まで間違えずに回す。train.py を実行する前、run 名や config を決めるとき、学習を再開・継続するとき、学習結果を記録・報告するときに使う。checkpoint の上書き事故、無視される設定、既存経路の破壊を防ぐ。
---

# 学習実験の回し方

`train.py` を叩く前に、この手順をそのまま順に実行する。各段の判定材料が揃わないうちに次へ進まない。

## 0. 前提（省略しない）

1. [leapsinger-verify](../leapsinger-verify/SKILL.md) が PASS していること。していないなら先に回す。
2. `git status` がクリーンか、変更が意図したものだけであること。**実験は commit された状態から始める。**
   後から「どのコードで出た結果か」を復元できないと、その run は捨てることになる。
3. データ契約を満たしていること。SVC は `data/<db>/{metadata.json, svc_shard.npz}` で、
   `content` `f0_interp` `uv` `loudness` `mel` の `T` が**完全一致**していること。
   loader は暗黙に直さず例外にする。これは前処理ミスを早期に出すための設計なので緩めない。

## 1. run 名を決める

**`--run_name` を再利用すると `log/<run_name>/ckpt_*.pt` の最新から黙って自動再開する。**
これが最も起こりやすい事故。別の実験なら必ず別名にする。

```bash
ls log/                      # 既存の run を必ず確認してから決める
```

命名は `<arch>_<データ>_<変えた要素>_<連番>` のように、後から差分が分かる形にする
（例: `svc_target_lr8_01`）。継続のつもりで同名を使うのは正しいが、**そのときは
「継続である」と明示的に判断したことを記録に残す**。

## 2. config を用意する

- 公開 config（`configs/svc_base.yaml` / `3speaker_gan2d.yaml` / `3singer_ritsu3style_uv_gan2d.yaml`）
  を直接編集しない。コピーして実験用の名前を付ける。
- `configs/.gitignore` は top-level の `configs/*.yaml` を無視する。**新しい config はコミットされない。**
  実験記録として残すなら、config の完全なコピーを成果物側に置く。
- `mel` セクションは前処理・loader・励起 hop で共有される。**前処理時と 1 つでも違うと無音や崩れになる。**
  44.1 kHz / hop 256 / n_fft 2048 / 128 mel / 40–16000 Hz（NHVSing V3 互換）から動かさない。

### 無視される・失敗する設定（先に潰す）

| 設定 | 実際の挙動 |
|---|---|
| `accum_steps` | **実装されていない。** 値を書いても無視される。実効 batch は増えない |
| `pitch_aug: true`（SVC） | `train.py` が SystemExit する。augmentation は特徴抽出の前に行う |
| `data.eval_songs`（曲が 1 つだけ） | `n_hold = min(eval_songs, 曲数-1)` で eval が空になる。評価は自動で飛ぶ |
| `--init_from` で SVS → SVC | 同一構造前提。arch をまたぐ部分ロードは未実装 |

## 3. 走らせる — **手元の GPU では回さない**

**決定:** 学習は vast.ai の Linux インスタンスで行い、手元の Windows 機は開発・推論・検証に使います。
手元で `--device cuda` の学習を始めると hook が止めます（`check_local_gpu_training`）。
理由は 2 つあります。実験記録の環境が本番と食い違うこと、そして他の作業と GPU を取り合って
`unspecified launch failure` のような一過性の失敗を起こすこと（実際に起きました）。
GPU が要るときは [vast-instance](../vast-instance/SKILL.md) に従ってインスタンスを用意します。


```bash
uv run python -m train --config <config> \
  --data_dirs <data...> --run_name <run> --out_root log --device cuda
```

長時間になるなら `run_in_background` で回し、通知を待つ。**sleep でポーリングしない。**

## 4. 走り出しを確認する（最初の 100 step で必ず見る）

1. `model <arch> <N>M params` が期待どおりか（SVC なら `harmonic_svc`）。
2. `[dataset]` / `[svc-dataset]` の phrase 数が期待どおりか。0 なら split か `min_sec` を疑う。
3. `train/flow` と `train/recon` が下がっているか。NaN や発散なら即止める。
4. GAN を使うなら `gan_start_step` を超えてから `train/d_loss` `train/adv` が出ているか。
5. **実測値を取る**: examples/sec、frames/sec、peak VRAM、checkpoint サイズ、eval 所要。
   これは [`doc/svc-data-compute.md`](../../../doc/svc-data-compute.md) の見積もりを実測へ置き換える材料であり、
   vast.ai なら**そのまま料金の見積もり**になる。

## 5. 成果物を守る

- vast.ai インスタンスの**ディスクは揮発する**。`log/<run>/ckpt_*.pt` と `events.out.tfevents.*` は
  実験の一部なので、走らせっぱなしにせず定期的に外へ退避する。
- base checkpoint は上書きしない。fine-tune は必ず別 `--run_name` / 別ディレクトリ。

## 6. 記録する（この run について何が言えるかを決める）

最低限これを残す。欠けると後で比較できない。

- Git commit と（あれば）dirty diff、config の完全なコピー
- dataset manifest / split / checksum
- `uv.lock` と `.python-version`、CUDA / driver / GPU
- seed、batch frames、peak VRAM、init checkpoint と load report
- train/validation 曲線、生成サンプル、失敗例のリスト
- best checkpoint の選び方、途中再開の履歴

## 7. 主張の範囲を守る

結果を書くときは [`doc/svc-prior-art-license.md`](../../../doc/svc-prior-art-license.md) 6 節の規則に従う。

- 「確認済み」はコード・実行 artifact・一次資料のいずれかを示せるときだけ。
- 「Seed-VC より良い」は同一 test set の blind comparison の後だけ。
- 「リアルタイム」は対象ハードでの end-to-end 遅延実測と連続動作の後だけ。
- 「世界初」「唯一」は使わない。「1-step」は acoustic flow の step 数であって pipeline 全体ではない。

完了したマイルストーンがあれば [`doc/svc-plan.md`](../../../doc/svc-plan.md) の判定材料と照合し、
[`doc/svc-implementation-status.md`](../../../doc/svc-implementation-status.md) の検証済み/未検証境界と
[`doc/svc.md`](../../../doc/svc.md) の完了レベルを**同時に**更新する（[leapsinger-docs](../leapsinger-docs/SKILL.md)）。
