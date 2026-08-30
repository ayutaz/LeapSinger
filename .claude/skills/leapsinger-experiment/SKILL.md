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

### 既存の成果物（上書きしないこと）

| run | 中身 | 使い方 |
|---|---|---|
| `m3_base` | **M3 の multi-singer base**。23 話者 / 約 18 時間 / 30,000 step。`ckpt_030000.pt` 134.7 MB | M4 の `--init_from` の入力。**別の `run_name` で fine-tune する** |
| `svc_seed0` / `svc_seed1` | 256 次元部分集合の A/B（3,000 step） | 比較の記録。再利用しない |

M3 の base は `spk_map` に 23 話者を持ちます（波音リツは **spk_id 22**、3 音源が同じ id）。
config は `log/m3_base/config.yaml` が**その run の完全な記録**です（`configs/svc_base_multi.yaml`
は `spk_map` を持たない recipe で、`tools/m3_corpus.py --write-config` が素材から埋めます）。

## 2. config を用意する

- 公開 config（`configs/svc_base.yaml` / `configs/svc_base_multi.yaml` / `3speaker_gan2d.yaml` /
  `3singer_ritsu3style_uv_gan2d.yaml`）を直接編集しない。コピーして実験用の名前を付ける。
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
| CUDA 推論の再現性 | **既定では bit 再現しない。** RNG ではなく conv/matmul の非決定性（実測 ln-mel で max 7.95e-02）。`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` で bit 一致する（CPU は既定で一致） |
| 無声だけの phrase | 曲を固定長で切るとイントロが丸ごと無声になる（実測 89 中 35 件）。**学習に入れると「無音を出す」ことを学ぶ。** `preprocess.svc.run --min-voiced` で除外する |
| `eval_items`（多話者） | **話者ごとの本数**。3 話者なら 9 サンプルだが、**23 話者だと 69 サンプル**になり、1 回の eval が mel 図と音声を 138 本書き出して **10 分以上**かかる（実測。単一コア 100%）。多話者では `eval_items: 1` と `eval_interval` を大きくする |

## 3. 走らせる — **手元では回さない（CPU も含めて）**

**決定:** 学習はすべて vast.ai の Linux インスタンスで行います。手元の Windows 機は開発・推論・
検証に使います。手元で `train.py` を起動すると **device によらず** hook が止めます
（`check_local_training`）。

**`--device cpu` に逃げてはいけません。** 実測で 1 phrase 1200 step に約 60 分かかり、
実験記録の環境も本番と食い違います。所要が短くても、遅くても、ローカルで済ませようとしないこと。
ローカル GPU は他の作業と取り合って `unspecified launch failure` を起こしました（実際に発生）。

インスタンスは [vast-instance](../vast-instance/SKILL.md) に従って用意します。


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
5. **実測値を取る**: `train.py` が step 100 / 1,000 / 10,000 で `[perf]` 行を出し、
   `log/<run>/perf.json` と TensorBoard の `perf/*` に残す（steps/sec、examples/sec、
   frames/sec、peak VRAM）。checkpoint サイズと eval 所要は別途控える。
   これは [`doc/svc-data-compute.md`](../../../doc/svc-data-compute.md) の見積もりを実測へ置き換える材料であり、
   vast.ai なら**そのまま料金の見積もり**になる。

   **実測の例（2026-08-30、RTX 3090 / 23 話者の multi-singer base）:** 8.14 step/s、
   130 ex/s、151k frames/s、**peak VRAM 1.95 GB**。見積もり表の「multi-singer base は 24 GB」は
   大きく外していた。`max_batch_size: 16` が先に効いて 1 batch が約 22,000 フレームで
   頭打ちになるため。**VRAM ではなく batch 本数が制約になることがある。**

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

**M3 で実際に回収したもの（同じ粒度で残す）:** `ckpt_030000.pt`、`config.yaml`、`perf.json`、
TensorBoard の events、**話者 25 分の `manifest.json` / `metadata.json`**、`m3_corpus.json`（素材の
選定）、抽出と学習のログ、検証の JSON と WAV、`nvidia-smi` の出力。**インスタンスのディスクは
破棄で消えるので、`destroy` の前に必ず回収する。**

## 6b. 推論の条件を学習と揃える（M3 で外した）

**学習（`preprocess.svc.run`）は生の音量のまま特徴を取ります。** 推論側で波形を加工すると、
`features_to_item()` の保証の**外側**で条件がずれます。実際に検証ツールが入力を peak 0.95 へ
正規化しており、波音リツ DB（peak 0.107）では**実質 19 dB の増幅**になって loudness 条件が
学習分布の 1.2σ 外へ出ていました。モデルは低域を持ち上げ高域を削り、spectral centroid が
620 → 368 Hz に落ちます。

- 推論の前に **loudness の正規化後の値**を学習分布（manifest の `loudness_mean` / `loudness_std`）と
  突き合わせる。±1σ を大きく外れていたら入力の扱いを疑う。
- **resample 以外の加工をしない。** 試聴用に音量を揃えるのは、特徴抽出の後で。
- **ただし持ち込み音源は例外。** 配信用に整えられた音源は peak 1.0 付近まで上げられており、
  学習素材（波音リツ DB は peak 0.107）から大きく外れます。実測で **+1.40σ**、
  spectral centroid が上限比 **−47%** まで落ちました。`svc_convert.py --match-loudness` で
  学習分布の平均へ寄せると **−24%** に戻ります。**「触らない」と「学習分布へ合わせる」は別**で、
  前者が既定、後者は明示的に指定するもの。

## 6c. 内容指標だけで判断しない

**content cos / F0 相関 / V/UV は高域の欠落を検知しません。** 上の不具合で centroid が
620 → 368 Hz に落ちても content cos は 0.8217 → 0.8096 しか動きませんでした。
[`tools/audio_metrics.py`](../../../tools/audio_metrics.py) の spectral centroid と帯域比を
併せて見ること。基準は source ではなく **「GT mel をボコーダーに通した再合成」（上限）**です。

**そして人に聴いてもらうこと。** この 2 件は利用者の「音がこもっている」という一言から
見つかりました。指標が揃って良いことを「問題なし」の根拠にしないでください。

## 7. 主張の範囲を守る

結果を書くときは [`doc/svc-prior-art-license.md`](../../../doc/svc-prior-art-license.md) 6 節の規則に従う。

- 「確認済み」はコード・実行 artifact・一次資料のいずれかを示せるときだけ。
- 「Seed-VC より良い」は同一 test set の blind comparison の後だけ。
- 「リアルタイム」は対象ハードでの end-to-end 遅延実測と連続動作の後だけ。
- 「世界初」「唯一」は使わない。「1-step」は acoustic flow の step 数であって pipeline 全体ではない。

完了したマイルストーンがあれば [`doc/svc-plan.md`](../../../doc/svc-plan.md) の判定材料と照合し、
[`doc/svc-implementation-status.md`](../../../doc/svc-implementation-status.md) の検証済み/未検証境界と
[`doc/svc.md`](../../../doc/svc.md) の完了レベルを**同時に**更新する（[leapsinger-docs](../leapsinger-docs/SKILL.md)）。
