# LeapSVC 実行計画（マイルストーン）

最終更新: 2026-08-30

対象ブランチ: `feature/svc`

## 0. この文書の位置づけ

他の文書は「何を作るか（要件・設計）」と「どう測るか（評価）」を扱います。この文書は **「どの順で進め、どこで止まったと判断するか」** だけを扱う独立した実行計画です。

既存文書には進行の見方が 4 通り併存しています。この計画はそれらを 1 本のマイルストーン列 M0〜M6 に統合したもので、対応関係は [8. 既存文書との対応](#8-既存文書との対応) にまとめます。

- [`svc.md`](svc.md) の完了レベル 1〜5（何が達成済みかの外向き宣言）
- [`svc-training.md`](svc-training.md) の Phase 0〜5（学習手順）
- [`svc-implementation-status.md`](svc-implementation-status.md) の「次の実装順序」7 項目（コード作業）
- [`svc-requirements.md`](svc-requirements.md) の Gate A/B/C（品質ゲート）

記述の確度ラベルは [`svc.md`](svc.md) の体系（**確認済み / 決定 / 推奨 / 見積もり / 仮説 / 未実装 / 要ユーザー判断**）に従います。

## 1. 現在地

**確認済み:** M0〜M6 のいずれも未完了です。到達しているのは前提条件にあたる実装と合成テンソル検証までで、これは完了レベル 1〜2 に相当します。

| 状態 | 内容 |
|---|---|
| 済 | `HarmonicSVCModel`、`ContentAdapter`、`SVCFeatureDataset`、train/infer 配線、`configs/svc_base.yaml` |
| 済 | 合成テンソルによる targeted test（shape、padding、alignment 検証、checkpoint 往復） |
| 済 | 再現可能な開発環境（Python 3.13 固定、`uv.lock`、CUDA 版 torch の実機疎通確認） |
| 済 | 全経路の疎通確認を 1 コマンド化（`tools/smoke/run_smoke.py` 11 ステージ）、誤コマンドを実行前に止める hook、作業手順の skill 化 |
| 済 | 学習環境の決定（vast.ai の Linux GPU）と、その操作系（`tools/vast.py` / `tools/vast_bootstrap.sh`） |
| 済 | content encoder・F0 extractor・loudness 正規化の決定（[content encoder の選定](svc-content-encoder.md)） |
| 未 | 実音声を 1 度も通していない。WAV も 1 本も出していない |

**決定:** 実音声を通していない段階では、品質・速度に関する対外的な主張を行いません。

## 2. マイルストーン一覧

| ID | 名前 | 一言でいう目的 | 完了の判定材料 |
|---|---|---|---|
| **M0** | データ確定 | 学習してよい素材を法的・品質的に固定する | dataset ledger と split list |
| **M1** | 特徴抽出前処理 | WAV から `svc_shard.npz` を再現可能に作る | 先に書いた失敗するテスト、再実行で一致する shard と manifest |
| **M2** | 実音声 smoke | 配線が実音声で成立することを示す | overfit した phrase の WAV |
| **M3** | multi-singer base | 話者に依存しない変換器の土台を作る | 未知 source singer で崩れない base ckpt |
| **M4** | target fine-tune | target singer の音色を再現する | offline teacher ckpt と指標一式 |
| **M5** | offline 品質ゲート | 外部 baseline と同条件で比較する | Seed-VC との blind test 結果 |
| **M6** | streaming student | 実機で実時間動作させる | 実測 end-to-end latency と連続運転記録 |

条件付きの派生作業は [7. 条件付きトラック](#7-条件付きトラック) に分離しました（SVS→SVC warm-start、NHVSing fine-tune、GAN）。これらは M0〜M6 の直列経路には入れません。

```text
M0 --> M1 --> M2 --> M3 --> M4 --> M5 --> M6
                |            |      |
               (A)          (B)    (C)     <- 条件付きトラック
```

---

## M0. データ確定

### 目的

学習・評価・配布のすべてが素材の権利と品質に依存します。ここを曖昧にしたまま先へ進むと、後段の成果物ごと使えなくなるため、最初に固定します。

### ゴール

次の 4 つが揃った状態を完了とします。

1. target singer と multi-singer corpus について、**学習可否・fine-tune 可否・重み配布可否・生成音声の利用条件**を規約 URL と取得日つきで記録した dataset ledger がある。
2. 全素材について clipping、伴奏漏れ、強い reverb、sample rate 誤りの検査を通し、除外したものは reject reason つきで残してある。
3. 音域（low/mid/high の滞在時間）と発声スタイル（chest / falsetto / breathy / 強声 / 弱声）の coverage を集計してある。
4. train / validation / test を**曲単位・収録セッション単位**で分離した split list があり、seed が記録されている。未知 source singer の test set が別に用意されている。

### 成果物

dataset ledger（権利・lineage・checksum）、split list、reject list、coverage 集計。

### 前提・依存

なし（起点）。

### 打ち切り・巻き戻し条件

権利条件を満たす target singer が確保できない場合、M1 以降を開始しません。**要ユーザー判断:** 商用利用の有無、配布形態（target 固定モデルのみか base も出すか）はここで確定させます。

---

## M1. 特徴抽出前処理

### 目的

**未実装:** WAV から `svc_shard.npz` を作るコードは現在リポジトリに存在しません。SVC 学習を 1 度も回せない唯一かつ最大のボトルネックであり、ここを埋めることが最優先です。

同時に、この段階で決めた抽出条件は以降のすべての実験の比較基盤になります。後から変えると M2 以降の結果が全部無効になるため、再現性の記録をゴールに含めます。

### ゴール

**実装は TDD で行います**（[進行ルール](#9-進行ルール)）。以下の 1〜4 は、実装より先に書いて失敗させたテストで担保されている状態を指します。

1. `preprocess` 配下のコマンド 1 本で、WAV ディレクトリから `data/<db>/{metadata.json, svc_shard.npz}` が生成できる。
2. 生成した shard を `SVCFeatureDataset` が**例外なく**読める。`content` `f0_interp` `uv` `loudness` `mel` の `T` が完全一致している（loader は暗黙補正しないので、ここが通ることが整合性の証明になります）。
3. phrase 名が `{song}_{NNNN}` 形式である。`_song_of()` による曲単位 train/eval 分割が効かなくなるため、この命名を崩さない。
4. 同じ WAV・同じ設定で 2 回実行し、出力が **bit 一致**する。
5. manifest に次を記録している: content encoder の **model revision と層番号**、sample rate、hop、**SSL の stride と mel grid への補間方法**、正規化方法、**256 次元部分集合の index と生成 seed**、loudness の定義（窓幅・floor・正規化単位）、F0 extractor の version、入力 WAV の checksum。
6. 重いモデル（ContentVec / RMVPE）を落とさずに走る単体テストと、実モデルを使う統合テストが分かれている。前者が `test_*.py` の既定、後者は明示的に指定したときだけ動く。

### 決定（[content encoder の選定](svc-content-encoder.md)）

| 項目 | 決定 |
|---|---|
| content encoder | ContentVec（`lengyue233/content-vec-best`、MIT、768 次元、layer 12）を凍結して使う |
| 学習時の次元 | 768 から固定ランダムに選んだ **256 次元**。shard には 768 を保存し、index で切り出す |
| F0 extractor | **RMVPE に固定** |
| loudness 特徴量 | **dataset 統計で正規化**（phrase 単位ではない） |
| 音声側の正規化 | RIFT-SVC に倣い **-18 LUFS** を検討。loudness 特徴量の正規化とは別レイヤ |

### 成果物

抽出コマンド、preprocessing manifest、target singer 1 人分の shard。

### 前提・依存

実行環境（Python 3.13 + CUDA 版 torch、`uv.lock`）と疎通確認は整備済みです。

**M0 との関係を分けます。** 抽出器の**実装とテストは M0 の完了を待たずに始められます**（テストは合成音声と小さな fixture で書けるため）。一方 M1 の**完了**（target singer 1 人分の実 shard）は M0 を要します。この分離により、権利確認と並行してコードを進められます。

### 残る未決

- **SSL 特徴を mel grid へ合わせる補間方法。** SSL は 16 kHz・stride 320 = 50 Hz、mel grid は 44,100/256 = 172.265625 Hz で、比 3.4453125 は整数になりません。**推奨:** まず linear。決めたら manifest に記録します。
- **256 次元部分集合の seed。** **推奨:** seed を 2 つ作り M2 の overfit で差を見ます（抽出は 768 を 1 回で済むので追加コストは小さい）。

**注意:** SVC では online `pitch_aug` が使えません（`train.py` が SystemExit します）。augmentation を行うならこの段階、特徴量抽出の前に行います。RMVPE はマルチプロセスで動かさないこと。

---

## M2. 実音声 smoke

### 目的

完了レベル 2（合成テンソル）と 3（実データ）の間を埋めます。**この段階の成功は音質も一般化も意味しません。** loader → condition → flow → vocoder の実配線が、実際の音で最後まで通ることだけを確認する工程です。

### ゴール

1. 1〜数 phrase を train set とし、loss が十分低下するまで overfit できる。
2. checkpoint から mel を生成し、NHVSing で WAV にできる。**完了レベル 3 の到達判定はこの WAV の存在**とします。
3. 出力 WAV を聴き、F0 の追従、無声区間の位置、長さが入力と整合している。
4. 同じ入力・seed・checkpoint で再実行し、結果が再現する。
5. 学習時と推論時で特徴量の正規化が同一であることを検証してある。

### 成果物

overfit run の log、生成 WAV、再現性の確認記録。

### 前提・依存

M1 完了。

### 打ち切り・巻き戻し条件

WAV が出ない、または明らかに入力と対応しない場合、原因が前処理（M1）にあるか配線にあるかを切り分けるまで M3 へ進みません。ground-truth mel を NHVSing に直接通した音と比較すれば、音響モデルの誤差と vocoder の誤差を分離できます。

---

## M3. multi-singer base pretraining

### 目的

target singer 単独で学習すると、SSL content に残った話者性を利用する近道を学んでしまい、**未知の source singer を入れたときに崩れます**。それを避けるため、複数歌手の自己再構成を speaker 条件つきで学習し、content / pitch / loudness と話者性が分離した土台を作ります。

### ゴール

1. 複数歌手の shard で `model.n_speakers` / `data.spk_map` / `train.balance_speakers` を設定し、speaker-balanced sampling で学習が完走する。
2. base checkpoint を **immutable な入力として保存**してある（M4 以降が上書きしない別ディレクトリ）。
3. 学習済みでない source singer を入れたとき、歌詞内容が崩壊しない。
4. 最初の 100 / 1,000 update で examples/sec、frames/sec、peak VRAM、checkpoint size、validation time を実測し、[データセットと計算資源](svc-data-compute.md) の見積もりを実測値へ更新してある。

### 成果物

base checkpoint、学習曲線、実測したスループットと VRAM、更新済みの見積もり表。

### 前提・依存

M2 完了、および M0 で multi-singer corpus の権利が確定していること。

### 規模の目安

**見積もり:** 20〜50 人 / 合計 100〜300 時間が現実的な最初の本学習案です。ただしこれは実測前の計画値であり、より小さい corpus で先に recipe を確定させます。

### 注意

**確認済み:** `train.py` は gradient accumulation を実装していません。config に `accum_steps` があっても無視されます。実効 batch を増やす計画を立てる場合、accumulation の実装とテストが先に必要です。

**決定:** この段階では GAN を無効のままにします（`gan.enabled: false`）。flow + 再構成が成立し、artifact を分類できる状態になってから、別 run で GAN を導入します（条件付きトラック C）。

---

## M4. target fine-tune（offline teacher）

### 目的

base の汎化を保ったまま、target singer の音色を再現します。この成果物が、以降のすべての比較と蒸留の基準（teacher）になります。

### ゴール

1. base checkpoint から target 専用の run directory で fine-tune し、base を上書きしていない。
2. **target similarity と、未知 source の明瞭度の両方**を記録してある。train 再構成だけが改善して未知 source が悪化する場合は早期停止する。
3. held-out song で変換した出力がある。
4. best checkpoint を train loss だけで選んでいない。選択規則が記録されている。
5. 各 run について実験記録が揃っている: Git commit と dirty diff、config の完全コピー、dataset manifest と split、依存 lock と CUDA / driver / GPU、seed、init checkpoint、failure list。

### 成果物

offline teacher checkpoint、指標一式、failure list、実験記録。

### 前提・依存

M3 完了。

### 規模の目安

**見積もり:** target singer 5〜10 時間が実用品質を狙う最初の推奨範囲です。ただし時間より coverage が重要で、5 時間あっても中音域の同じ歌い方だけなら高音裏声や強い子音で崩れます。

### 未決

**要ユーザー判断:** speaker embedding を残して target ID を固定するか、embedding を焼き込んで speaker 入力のないモデルにするか。**推奨:** base 学習中は話者条件を残し、固定化は fine-tune と配布の段階で判断します。

---

## M5. offline 品質ゲート（Seed-VC 比較）

### 目的

**内部の loss ではなく外部の baseline に対して**、同一条件で品質を示します。ここを通過するまで、リアルタイム化の最適化を主目的にしません。

### ゴール

Gate B の全項目を満たすこと。

1. held-out song と未知 source singer の両方で変換できている。
2. 客観指標を記録してある: F0 correlation / RMSE / V-UV error、speaker similarity、CER、信号品質、timing、RTF と peak VRAM（特徴抽出と vocoder を含むか除くかを併記）。
3. Seed-VC と **blind listening test** を実施してある。system 名を隠し、順序を randomize し、loudness を揃えてある。評価者数・clip 数・除外規則を事前に決めてある。
4. failure sample を除外せず、[評価計画](svc-evaluation.md) の failure taxonomy で分類して残してある。
5. Seed-VC 側の revision、checkpoint、inference steps、F0 条件、reference clip を manifest に固定してある。default と高速設定を混同しない。

### 成果物

評価 report（[評価計画](svc-evaluation.md) 9 節のテンプレート準拠）、全評価 clip の manifest と出力、blind test の生データ。

### 前提・依存

M4 完了。

### 主張の制約

**決定:** 「Seed-VC より良い」という表現は、このマイルストーンの blind comparison を経た後にのみ使います。ここを通過して初めて完了レベル 4 到達です。

**注意:** Seed-VC は GPL-3.0 です。外部 baseline として実行するだけとし、コードをこのリポジトリへ取り込みません。比較 script と manifest は独立に管理します。

### 要ユーザー判断

合格とする最小差、評価者数、対象楽曲の閾値は、test set 作成後に確定します。

---

## M6. streaming student

### 目的

offline teacher は phrase 全体を見る non-causal モデルです。実時間で使うには未来の文脈を削る必要があり、その劣化を蒸留で最小化します。

### ゴール

Gate C の全項目を満たすこと。

1. teacher と**同じ test set**で student を評価し、品質低下と遅延改善を同時に報告してある。
2. chunk 境界に click、音切れ、F0 jump がない。
3. 実機で測定してある: RTF、algorithmic lookahead、そして audio I/O を含む **end-to-end latency**。内訳（input buffer / SSL encoder の receptive field / pitch extractor の lookahead / student の lookahead / vocoder の overlap / output buffer）を分解して記録する。
4. 長時間連続運転で buffer overrun / underrun が発生しない。

### 成果物

student checkpoint、遅延内訳表、連続運転ログ、teacher との比較 report。

### 前提・依存

M5 通過。**決定:** M5 を通過していない teacher を基準に student を作りません。

### 主張の制約

**決定:** 「リアルタイム」は、対象ハードウェアでの end-to-end latency 実測と連続動作の確認後にのみ使います。model の RTF 単体では根拠になりません。README 記載の RTF 0.027 は既存 **SVS 音響モデル**の値であり、SVC の特徴抽出・ボコーダー・ストリーミング I/O を含みません。

### 要ユーザー判断

許容する lookahead と往復遅延、音質と遅延のどちらを優先するか。

---

## 7. 条件付きトラック

直列経路には入れず、条件が成立したときだけ実施します。

### (A) SVS → SVC warm-start

**目的:** 既存 SVS checkpoint の flow backbone と励起を再利用し、M3 の学習コストを下げられるかを確認する。

**ゴール:** 部分 weight mapping と missing/unexpected key の allowlist を持つ loader、ロード結果の記録、そして**独立初期化した baseline との比較結果**。速度・品質の利得が示せない場合は採用しません。

**前提:** M2 完了。**未実装:** 現在の `--init_from --finetune` は同一構造の checkpoint を前提としており、SVS→SVC の部分ロードには使えません。

**実施判断:** M3 の学習コストが問題になった場合のみ。再利用できるのは flow backbone・励起・損失・discriminator・NHVSing interface で、phoneme embedding・length regulation・phoneme encoder の入力 projection は再利用できません。

### (B) NHVSing target fine-tune

**目的:** vocoder 側に起因する artifact を切り分けて解消する。

**ゴール:** ground-truth mel を入力しても同じ artifact が出ることを示したうえで、fine-tune 前後を比較した結果。

**実施条件:** 次の 3 つがすべて成立したときのみ実施します。

1. target singer の高音や裏声で vocoder artifact が系統的に出る。
2. ground-truth mel を入力しても同じ artifact が出る。
3. 音響モデルの誤差と vocoder の誤差を切り分けられている。

**前提:** M4 完了。まず既存重みを固定した baseline を作ってから比較します。**注意:** mel 分布への過適合、データ規約、重み配布条件を別途確認します。NHVSing V3 の配布重みは非商用データセットで学習されています。

### (C) GAN の導入

**目的:** 質感の鮮明化。

**ゴール:** GAN なし baseline との A/B 比較。

**実施条件:** flow + 再構成が安定し、artifact を分類できる状態になってから、**別 run** で導入します。既存 SVS と同様に `gan_start_step` 以降の二段構成とします。

**注意:** GAN 経路（`_forward_flow_gan`）は `compute_loss` と数式・mask・RNG 順を一致させて再実装されています。片方だけ変更すると学習が一致しなくなります。

---

## 8. 既存文書との対応

| 本計画 | 完了レベル（[svc.md](svc.md)） | Phase（[svc-training.md](svc-training.md)） | Gate（[svc-requirements.md](svc-requirements.md)） | 次の実装順序（[status](svc-implementation-status.md)） |
|---|---|---|---|---|
| （前提・到達済） | 1 実装 / 2 合成 smoke | — | — | — |
| M0 | — | — | **Gate A** | — |
| M1 | — | Phase 0 前半 | — | 1, 2 |
| M2 | **3 実データ** | Phase 0 後半 | — | 3 |
| M3 | — | Phase 1 | — | 5 前半 |
| M4 | — | Phase 2 | — | 5 後半 |
| M5 | **4 品質比較** | Phase 4 | **Gate B** | 6 |
| M6 | **5 リアルタイム** | Phase 5 | **Gate C** | 7 |
| (A) | — | 5 節 warm-start | — | 4 |
| (B) | — | Phase 3 | — | — |

## 9. 進行ルール

- **飛ばさない。** M1 を経ずに M3 以降は実行できません（shard がないため物理的に不可能）。M5 を経ずに M6 を主目的にしません。
- **上書きしない。** base checkpoint と fine-tune 成果物は別 `run_name` / 別ディレクトリに保存します。同じ `--run_name` で再実行すると `log/<run_name>/ckpt_*.pt` の最新から自動再開するため、別実験では必ず run 名を変えます。
- **テストを先に書く。** このリポジトリの実装はすべて TDD で行います。失敗するテストを書き、失敗を確認し、通す最小限のコードを書く。先に書いたテストが無い実装コードは破棄してやり直します（`superpowers:test-driven-development` と `leapsinger-tdd`）。
- **1 度に 1 要素。** ablation では同一 split・seed・更新 budget を使い、複数要素を同時に変えません。
- **環境を固定する。** Python 3.13 と `uv.lock` を実験の一部として扱い、torch / CUDA / 依存を更新した run はその差分を実験記録に残します。環境差による品質差はモデル差と見分けがつきません。
- **失敗を消さない。** failure clip は削除せず、category と suspected component を付けて残します。
- **完了を宣言する条件。** 各マイルストーンは、コードの存在ではなくゴール節の判定材料が揃った時点で完了とします。完了時に [`svc-implementation-status.md`](svc-implementation-status.md) の検証済み / 未検証境界と [`svc.md`](svc.md) の完了レベルを同時に更新します。
- **主張の範囲。** [先行研究・ライセンス・リスク](svc-prior-art-license.md) 6 節の主張ルールに従います。「世界初」「唯一」は使いません。「1-step」は acoustic flow の step 数であり、pipeline 全体の話ではありません。
