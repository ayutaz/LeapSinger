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

**確認済み:** **M0 / M1 / M2 / M3 は完了**、M4 以降は未着手です。
[svc.md](svc.md) の完了レベルは **3（実データ）に到達**しています（実音声の shard で学習して WAV を出した）。
レベル 4 は Seed-VC との blind comparison を要求するので未到達です。

| 状態 | 内容 |
|---|---|
| 済 | `HarmonicSVCModel`、`ContentAdapter`、`SVCFeatureDataset`、train/infer 配線、`configs/svc_base.yaml` |
| 済 | 合成テンソルによる targeted test（shape、padding、alignment 検証、checkpoint 往復） |
| 済 | 再現可能な開発環境（Python 3.13 固定、`uv.lock`、CUDA 版 torch の実機疎通確認） |
| 済 | 全経路の疎通確認を 1 コマンド化（`tools/smoke/run_smoke.py` 12 ステージ）、誤コマンドを実行前に止める hook、作業手順の skill 化 |
| 済 | 学習環境の決定（vast.ai の Linux GPU）と、その操作系（`tools/vast.py` / `tools/vast_bootstrap.sh`） |
| 済 | content encoder・F0 extractor・loudness 正規化・補間方法・部分集合 seed の決定（[content encoder の選定](svc-content-encoder.md)） |
| 済 | **M0 完了**（入手可能な素材について）。5 コーパスの取得・検査・coverage・split と台帳（[データセット台帳](svc-dataset-ledger.md)） |
| 済 | **M1 完了**。WAV から shard までコマンド 1 本、再実行で bit 一致 |
| 済 | **M2 完了**。実音声 2 phrase を overfit して WAV を生成。F0 相関 0.9991、決定的モードで bit 再現 |
| 済 | **M3 完了**。23 話者 18 時間の base を **60,000 step** 学習（30,000 で完了判定 → その後 ① で継続）。未知 source で内容が崩壊しないことを実測 |
| 未 | **音質と target らしさを評価していない。** 内容保持・F0 追従・音の明るさは測りましたが、それは音質ではありません |
| 済 | **高域不足の原因を特定**（2026-08-31、40 clip で測り直し）。学習不足ではなく**入力の F0 に対する出力傾斜の強い結合**。男性 source を +12 半音すると 540 → 1196 Hz と明るさが戻る |
| 未 | **1 step 写像はほぼ収束**（60,000 step で 1 step と 16 step の偏差の絶対値が 34.3 対 34.0）。**継続学習でこれ以上 1 step 品質は改善しません** |

**決定:** 実データで汎化する学習を行うまで、品質・速度に関する対外的な主張を行いません。

**教訓（2026-08-30）:** M3 の完了を宣言した後、利用者が出力を聴いて「音がこもっている」と
報告したことから欠陥が 2 件出ました。**検証一式（content cos / F0 相関 / V/UV）は両方とも
素通りしていました。** 指標が揃って良いことを「問題なし」の根拠にせず、**人に聴いてもらう**
こと。詳細は M3 の進捗節。

## 2. マイルストーン一覧

| ID | 名前 | 一言でいう目的 | 完了の判定材料 |
|---|---|---|---|
| **M0** ✅ | データ確定 | 学習してよい素材を法的・品質的に固定する | dataset ledger と split list |
| **M1** ✅ | 特徴抽出前処理 | WAV から `svc_shard.npz` を再現可能に作る | 先に書いた失敗するテスト、再実行で一致する shard と manifest |
| **M2** ✅ | 実音声 smoke | 配線が実音声で成立することを示す | overfit した phrase の WAV |
| **M3** ✅ | multi-singer base | 話者に依存しない変換器の土台を作る | 未知 source singer で崩れない base ckpt（高域不足が残る。下記の分岐へ） |
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

次の 5 つが揃った状態を完了とします。

1. target singer と multi-singer corpus について、**学習可否・fine-tune 可否・重み配布可否・生成音声の利用条件**を規約 URL と取得日つきで記録した dataset ledger がある。
2. 全素材について clipping、伴奏漏れ、強い reverb、sample rate 誤りの検査を通し、除外したものは reject reason つきで残してある。
3. 音域（low/mid/high の滞在時間）と発声スタイル（chest / falsetto / breathy / 強声 / 弱声）の coverage を集計してある。
4. **base corpus が NHVSing にとって in-distribution かを判断してある。** 同梱ボコーダー NHVSing V3 は 本プロジェクトと同一の mel 仕様（44.1 kHz / hop 256 / 128-mel / 40–16000 Hz ln）で、特定の 10 コーパスで学習されています（[台帳](svc-dataset-ledger.md) 7 節）。そこに含まれない歌手の mel は vocoder にとって未知になり得ます。**ground-truth mel の再合成忠実度**を既知/未知のコーパスで比較して判断します。
5. train / validation / test を**曲単位・収録セッション単位**で分離した split list があり、seed が記録されている。未知 source singer の test set が別に用意されている。**性別で層化する**（実データで偏りを確認済み）。

### 成果物

dataset ledger（権利・lineage・checksum）、split list、reject list、coverage 集計。

### 進捗（2026-08-30）: 入手可能な素材について完了

**確認済み: 5 つのゴールすべてを実データで実行しました。** 詳細と数値は
[データセット台帳](svc-dataset-ledger.md) です。

| ゴール | 実施内容 |
|---|---|
| 1. dataset ledger | 既存 3 DB の権利条件を一次資料から記録。公開コーパス 14 件のライセンス調査。fork 元と同梱 NHVSing の学習データ。取得物の SHA-256。**素材の割り当てを確定** |
| 2. reject list | 5 コーパス全件に検査。理由つきで記録 |
| 3. coverage | RMVPE で実 F0 を抽出して音域帯の滞在時間。GTSinger は注釈から**音素単位**の技法 coverage |
| 4. NHVSing の in/out-of-distribution | **実測して判断済み**。ground-truth mel の再合成忠実度を 5 コーパス × 12 clip で比較し、**GTSinger にペナルティは見えない**（[台帳 7b 節](svc-dataset-ledger.md#7b-nhvsing-にとって-in-distribution-かの実測2026-08-30)） |
| 5. split list | 曲・歌手単位。**性別で層化**、**曲名を正規化**して leakage を排除 |

**確定した割り当て:**

| 役割 | 素材 | 実測 |
|---|---|---|
| target singer | **波音リツ 3 音源** | 10.41 h / 75 曲 / 除外 0 |
| multi-singer base | **GTSinger 全 9 言語 + 日本語 3 DB** | GTSinger 80.59 h（日本語分 6.79 h 検証済み） |
| 未知 source の test | **VocalSet** | 8.73 h / 20 歌手 |

**この過程で 10 件の欠陥・訂正が出ました。いずれも合成データでは出ないものです。**
代表例: 歌手 ID の取り違えと曲名正規化の欠如による **leakage の実検出**、
split の性別偏り、帯域閾値 16 kHz が実歌唱に対して厳しすぎたこと（推奨を撤回）、
「波音リツは低音」という**推測の誤り**（実測は p50 360 Hz で C4–C5 中心）。

**未取得:** 東北きりたんと No.7 はログイン必須で取得できませんでした（HTTP 応答で確認済み。
kiritan の公開 GitHub はラベルのみで音声を含みません）。**低音域の補強**に効く素材なので、
入手できたときに同じ検査を通します。

**未実装:** 発声スタイルの coverage は、**技法ラベルを持つコーパス（GTSinger / VocalSet）でのみ**
自動集計できます。ラベルの無い素材では手作業の注釈が要ります。

### 前提・依存

なし（起点）。

### 打ち切り・巻き戻し条件

権利条件を満たす target singer が確保できない場合、M1 以降を開始しません。

**決定（2026-08-30）:** 用途は**研究・個人利用のみ、モデルの配布なし**。これにより CC-NC 系も research-only のコーパスも使えます。素材の選定は [データセット台帳](svc-dataset-ledger.md) 6 節。**将来 配布や商用利用へ移る場合は、この決定と素材の選定をやり直す必要があります。**

---

## M1. 特徴抽出前処理

### 目的

WAV から `svc_shard.npz` を作れないと SVC 学習を 1 度も回せません。ここが直列経路の最大のボトルネックでした。

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
| 学習時の次元 | 768 から固定ランダムに選んだ **256 次元**（seed 0 を既定）。生の 768 は cache に、shard には 256 を書く |
| 補間 | SSL 50 Hz -> mel grid へ **left（直前保持・左寄せ繰り返し）**。先読み 0 ms、元の SSL ベクトルを保持 |
| F0 extractor | **RMVPE に固定** |
| loudness 特徴量 | **dataset 統計で正規化**（phrase 単位ではない） |
| 音声側の正規化 | RIFT-SVC に倣い **-18 LUFS** を検討。loudness 特徴量の正規化とは別レイヤ |

### 成果物

抽出コマンド、preprocessing manifest、target singer 1 人分の shard。

### 前提・依存

実行環境（Python 3.13 + CUDA 版 torch、`uv.lock`）と疎通確認は整備済みです。

**M0 との関係を分けます。** 抽出器の**実装とテストは M0 の完了を待たずに始められます**（テストは合成音声と小さな fixture で書けるため）。一方 M1 の**完了**（target singer 1 人分の実 shard）は M0 を要します。この分離により、権利確認と並行してコードを進められます。

### 抽出は 2 段に分ける

**決定:** 抽出器は次の 2 段構成にします。入口のコマンドは 1 本のままで、2 段目だけの再実行もできるようにします。

| 段 | 入力 -> 出力 | コスト | 内容 |
|---|---|---|---|
| 1. `extract` | WAV -> 生特徴 cache | 重い（GPU・モデル） | ContentVec 768 を 50 Hz のまま、RMVPE の F0、RMS、mel |
| 2. `build-shard` | cache -> `svc_shard.npz` | 軽い（CPU のみ） | mel grid への整列、dataset 統計での正規化、256 次元の切り出し |

**理由:** 下の 2 つの決定はどちらも「後で変えたくなる」種類のものです。2 段に分けておけば、**補間方法と 256 次元 seed の比較が 2 段目の再実行だけで済み、ContentVec と RMVPE を回し直さずに ablation できます。** vast.ai の課金に直接効きます。

shard に書くのは 256 次元です（768 ではありません）。loader に切り出しをさせると「暗黙に直さない」という契約の性質を崩すためです。

### 決定 1: 補間は left（直前保持）

**確認済み（実測）:** SSL は 16 kHz・stride 320 = 50 Hz、mel grid は 44,100/256 = 172.265625 Hz。比 3.4453125 は整数になりません。3 方式を同条件で測った結果:

| 方式 | left 基準の先読み | ブレンドされるフレーム |
|---|---:|---:|
| **left（直前保持）** | **0 ms** | **0%** |
| nearest | 20 ms | 0% |
| linear | 20 ms | 99.4% |

**決定:** **left** を採用します。理由は 3 つで、いずれも他方式より劣る点がありません。

1. **先読み 0 ms。** nearest と linear は 1 SSL フレーム（20 ms）先を読みます。[M6](#m6-streaming-student) は lookahead を削る作業なので、前処理の時点で無償の 20 ms を積むのは筋が悪く、しかも train と inference の因果性がずれます。
2. **元の SSL ベクトルをそのまま保持。** linear は 99.4% のフレームを混ぜます。ContentVec の表現空間が線形とは限らず、音素境界をまたぐ混合は実在しないベクトルを作ります。
3. **so-vits-svc の既定（`repeat_expand_2d` の `mode='left'`）と一致。** 最も広く使われている挙動です。

以前この文書では「推奨: まず linear」と書いていましたが、実測の結果 **left に訂正**しました。

### 決定 2: 256 次元は seed 0 を既定、M2 で seed 1 と比較

**決定:** 既定は seed 0。M2 の overfit で seed 1 との差を見ます。Interspeech 2025 の 2 つの部分集合は SSIM 0.813 / 0.822 と小さいながら差があったため、1 回だけ確かめます。2 段構成なので追加コストは 2 段目の再実行だけです。

**未実装だが選択肢として残す:** cache に 768 が残るので、ランダム部分集合ではなく PCA や学習可能な射影も後から試せます。

**注意:** SVC では online `pitch_aug` が使えません（`train.py` が SystemExit します）。augmentation を行うならこの段階、特徴量抽出の前に行います。RMVPE はマルチプロセスで動かさないこと。

### 進捗（2026-08-30）: 完了

**確認済み:** WAV ディレクトリから shard までコマンド 1 本で作れます。実装は
`preprocess/svc/{chunk,extract,encoders,shard,run}.py`、テストは `test_svc_preprocess.py`
（84 件。重いモデルを使わない）と `test_svc_preprocess_integration.py`（実 ContentVec / RMVPE、
既定 skip）です。

| ゴール | 結果 |
|---|---|
| 1. コマンド 1 本 | `uv run python -m preprocess.svc.run --wav-dir <dir> --out <db>`。波音リツ 1 曲を 42 秒で抽出 |
| 2. loader が読める | 実 `SVCFeatureDataset` が例外なく読むことをテストで固定。全配列の `T` が完全一致 |
| 3. phrase 名 `{song}_{NNNN}` | `song_name()` の出力が `_song_of()` を往復することをテストで固定 |
| 4. bit 一致 | 同じ cache・同じ設定で作り直して **sha256 が一致**（`np.savez` の zip 時刻を固定） |
| 5. manifest | encoder の id / revision / 層 / sr / stride、補間方法、部分集合の **index と seed**、loudness の窓幅・floor・正規化単位、**RMVPE 重みの sha256 と入手元**、入力 WAV の checksum |
| 6. テストの分離 | 単体は encoder を引数で受け取る fake で動く。実モデルは `LEAPSINGER_INTEGRATION=1` のときだけ |

**未実施:** 決定 2 の **seed 0 と seed 1 の比較**。shard は両方作ってありますが（`subset_seed`
は manifest に記録済み）、overfit で差を見る作業は行っていません。既定の seed 0 で M2 を通した
ため直列経路は止まっていません。

**決定（2026-08-30）: M3 のインスタンスで、base 学習を始める前に回します。** 専用インスタンスを
立てずに済み、追加コストは数分です。手順は M3 の「開始時にやること」を参照。

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

### 進捗（2026-08-30）: 完了

**確認済み:** vast.ai の RTX A4000 インスタンス（$0.098/hr、約 27 分、実費 約 $0.04）で
波音リツ 2 phrase（各 3 秒）を 3,000 step overfit し、WAV を生成しました。
検証は [`tools/m2_verify.py`](../tools/m2_verify.py)、報告は `m2_report.json` です。

| ゴール | 結果 |
|---|---|
| 1. overfit できる | flow 0.00710 → 0.00235（最小 0.00115、**3.0 倍**）、recon 0.03573 → 0.01889（最小 0.00193） |
| 2. **WAV を生成できる** | **`m2_pred.wav` 3.00 秒 / 44.1 kHz / peak 0.950。完了レベル 3 の到達判定を満たす** |
| 3. F0・無声・長さの整合 | 長さ 132,096 = 516 × 256 で一致。**F0 相関 0.9991・中央値 0.012 半音・p90 0.075 半音**、V/UV 一致率 0.988 |
| 4. 再現性 | 決定的モードで **bit 一致**（max diff 0.0） |
| 5. 学習と推論で正規化が同一 | `features_to_item()` が manifest の統計と部分集合を当て、**shard の値と 1 bit も違わない**ことをテストで固定 |

**切り分け:** ground-truth mel を NHVSing に直接通した WAV も出しました。GT 側の F0 相関 0.9990 /
中央値 0.012 半音に対し、予測側は 0.9991 / 0.012 半音で**ほぼ同等**です。したがって
**この phrase では音響モデルが vocoder の性能を損なっていません**。

#### この過程で見つかった欠陥 2 件

**1. 無声だけの phrase が学習に入っていた（M1 の欠陥）。** 最初の overfit で生成された WAV が
無音になりました。追うと入力の `uv` が全フレーム 0 で、曲の先頭 27 秒がイントロの無音だったためです。
固定長で先頭から切ると、**89 chunk 中 35 件（39%）が完全に無声**になっていました
（しきい値 0.5 で最終的に除外されたのは 39 件、採用は 50 phrase）。
`chunk.py` に `voiced_ratio()` を足し、抽出時に `--min-voiced`（既定 0.3）未満を捨てるようにしました。
除外後は有声率が最小 0.516 / 平均 0.857 になりました。**合成データでは出ない欠陥です。**

**2. CUDA 推論が bit 再現しない。** 切り分けたところ RNG ではなく CUDA の conv/matmul の
非決定性でした（CPU は bit 一致、CUDA は max 7.95e-02 / mean 4.33e-04 の差）。
`torch.use_deterministic_algorithms(True)` と `CUBLAS_WORKSPACE_CONFIG` を有効にすると
**CUDA でも bit 一致**します。検証スクリプトでは既定で有効にしました。

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

M2 完了、および M0 で multi-singer corpus の権利が確定していること。両方とも満たしています。

### 開始時にやること（base 学習の前に）

**決定:** M1 決定 2 の **seed 0 / seed 1 比較**をここで片づけます。インスタンスを立てた直後、
base 学習を始める前に、同じ split・同じ seed・同じ更新 budget で 2 本の overfit を回して
比較します。専用インスタンスを立てずに済み、追加は数分です。

```bash
# 2 段目だけを 2 通り回す（ContentVec / RMVPE の再実行は不要）
uv run python -m preprocess.svc.run --from-cache data/ritsu/_cache --out data/seed0 --subset-seed 0
uv run python -m preprocess.svc.run --from-cache data/ritsu/_cache --out data/seed1 --subset-seed 1
# run 名を分けて 2 本。base checkpoint は別 run_name（上書きしない）
uv run python -m train --config configs/svc_base.yaml --data_dirs data/seed0 \
  --run_name svc_seed0 --out_root log --device cuda --max_updates 3000
uv run python -m train --config configs/svc_base.yaml --data_dirs data/seed1 \
  --run_name svc_seed1 --out_root log --device cuda --max_updates 3000
uv run python tools/m2_verify.py --ckpt log/svc_seed0/ckpt_003000.pt --data data/seed0 --out out/seed0
uv run python tools/m2_verify.py --ckpt log/svc_seed1/ckpt_003000.pt --data data/seed1 --out out/seed1
```

差が無ければ seed 0 で確定し、以後の全実験の基盤にします。差が出たら、その差を
[content encoder の選定](svc-content-encoder.md) へ記録してから base 学習へ進みます。

### 素材の構成

**決定:** [台帳](svc-dataset-ledger.md) 6 節の割り当てに従い、**GTSinger 全 9 言語 20 歌手 +
日本語 3 DB** を使います。用意は [`tools/m3_corpus.py`](../tools/m3_corpus.py) が行います。

| 決定 | 内容 |
|---|---|
| 話者ごとに 1 shard ディレクトリ | `svc_dataset.py` は speaker id を**ディレクトリ名**から引く（`spk_map`）。話者を分けるにはディレクトリを分けるしかない |
| 波音リツ 3 音源は同じ speaker id | 同じ歌手の別の声質。SVS 側の config と同じ扱い |
| **歌手ごとに分量を揃える**（`--max-hours`） | base 事前学習で効くのは総時間より**話者の多様性**。GTSinger 80.59 h を全部抽出すると 12 時間以上かかる |
| GTSinger は使う wav だけ落とす | リポジトリは 149,037 ファイルあり、丸ごと落とすと HTTP 429 で律速されて 3 時間半かかる（実測） |
| 曲名は `--song-parts` で指定 | `<言語>/<歌手>/<技法>/<曲>/<Group>/NNNN.wav` の `<曲>`。技法や Group をまたいで同じ曲を同じ song 名にしないと leakage する |

`spk_map` と `n_speakers` は素材で決まるので `configs/svc_base_multi.yaml` には書かず、
`--write-config` が**実際に作った shard から**埋めた config を run ディレクトリへ書き出します。
**その生成物が実験記録**です。

### 進捗（2026-08-30）: 完了

**確認済み:** vast.ai の RTX 3090（実効 $0.21/hr）で **23 話者・25 shard・約 18 時間**の
multi-singer base を 30,000 step 学習し、4 つのゴールをすべて満たしました。
成果物は `log/m3_base/`（config・perf.json・events・checkpoint）と `out/m3_*`（検証）です。

| ゴール | 結果 |
|---|---|
| 1. balanced sampling で完走 | `balance_speakers: true` で `[dataset] balanced sampling by 'speaker'`。train 8,353 phrase / hold-out 888。**30,000 step 完走**。train/flow 0.04717 → **0.00551**（最小 0.00447）、recon 0.08960 → 0.02891。eval/loss 0.02932 → **0.02311（単調減少）**、eval/varL 1.200 → 1.064（1.0 = GT と同じ鮮鋭さ） |
| 2. base を immutable に保存 | `ckpt_030000.pt` **134.7 MB** を手元へ回収。ローカルで読み込み確認（`HarmonicSVCModel` 11.61M params / `n_speakers` 23 / `spk_bank` (23, 32)）。**M4 は別 `run_name` を使う** |
| 3. **未知 source で内容が崩壊しない** | 下表。**未知 source が学習済み歌手と同等** |
| 4. 実測して見積もりを更新 | 8.14 step/s・130 ex/s・151,354 frames/s・**peak VRAM 1.95 GB**・checkpoint 134.7 MB。[データセットと計算資源](svc-data-compute.md) 6 節・8 節を実測値へ更新済み |

#### ゴール 3 の測り方と結果

「歌詞内容が崩壊しない」を聴かずに測るため、**source と変換後の ContentVec の
フレームごと cos 類似度**を使いました（[`tools/m3_verify.py`](../tools/m3_verify.py)）。
単独の数値には意味が無いので、**上限**（source の GT mel を NHVSing に通しただけの再合成＝
vocoder のみの劣化）と**下限**（無関係な別クリップとの類似度）で挟みます。

| source | n | content cos | 上限 | 下限 | **下限からの回復率** | F0 相関 | 半音 | V/UV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **未知（VocalSet。base に不使用）** | 8 | 0.8217 | 0.9434 | 0.0923 | **85.7%** | 0.9914 | 0.018 | 0.9884 |
| 対照: 学習済み歌手（JA-Soprano-1） | 8 | 0.8124 | 0.9437 | 0.1035 | 84.4% | 0.9984 | 0.016 | 0.9840 |

**未知 source のほうがわずかに良い**という結果で、少なくとも「未知だから崩れる」現象は
出ていません。F0 追従（相関 0.99 以上・中央値 0.02 半音）と V/UV 一致（0.98 以上）も保たれています。

**この結果が意味しないこと:** 音質、target らしさ（話者類似度）、Seed-VC との優劣。
content cos は「内容が保たれているか」だけを見る指標で、**M4 / M5 の評価の代わりにはなりません**。
n=8・6 秒クリップという規模でもあります。

#### 試聴で見つかった欠陥 2 件（2026-08-30、M3 完了の宣言後）

**確認済み:** 利用者が出力を聴いて「音がこもっている」と報告したことから、次の 2 件が出ました。
**どちらも M3 の検証一式（content cos / F0 相関 / V/UV）を素通りしていました。**

**1. 推論ツールが入力を peak 正規化していた（学習と条件が違う）。** `preprocess.svc.run` は
生の音量のまま特徴を取りますが、`tools/m3_verify.py` と `tools/svc_convert.py` は
peak 0.95 へ揃えてから抽出していました。波音リツの DB は peak 0.107 と小さく、**実質 19 dB の
増幅**になります。loudness 条件が学習分布（mean −4.63 / std 1.69）の **1.2σ 外**へ出て、
モデルが低域を持ち上げ高域を削っていました。

| リツ自己再構成（8 秒） | mel L1 | 低域 bin 差 | 高域 bin 差 | centroid |
|---|---:|---:|---:|---:|
| 修正前（peak 正規化） | 0.727 | +0.439 | −0.636 | 368 Hz |
| **修正後（生の音量）** | **0.558** | +0.101 | −0.236 | **620 Hz** |

M2 ゴール 5「学習時と推論時で正規化が同一」は `features_to_item()` で保証していましたが、
**その手前で波形を加工していた**ので、保証の境界の外側で破れていました。両ツールを直し、
peak 正規化は `--peak-normalize` の明示指定でしか起きないようにしています。

**2. 検証指標が高域の欠落を検知しない。** この不具合で spectral centroid が 620 → 368 Hz へ
落ちても、**content cos は 0.8217 → 0.8096 としか動きませんでした**。F0 相関も V/UV も同様です。
内容・音高・有声区間だけを見ていると、音の明るさの劣化は素通りします。
[`tools/audio_metrics.py`](../tools/audio_metrics.py) に帯域エネルギー比と centroid を切り出し、
`m3_verify` の報告へ載せました（契約テスト 6 件）。

**M3 ゴール 3 の結論は変わりません。** 修正後に測り直しても content cos 0.799〜0.810 /
回復率 84% 前後で、バグの有無にほぼ依存しませんでした（指標が鈍感だったため）。

**残る鈍さは「モデルの限界」ではなく「1 step 写像の未収束」でした（2026-08-30 追測）。**

利用者の再度の指摘（「まだこもっている／つぶれている」）を受けて flow の step 数を振ったところ、
**step を増やすと明るさが戻ります**。上限（GT mel をボコーダーに通した再合成）との差は次のとおり。

| 区間 | 上限 | **1 step** | **16 step** |
|---|---:|---:|---:|
| 波音リツ 1st_color | 875 Hz | 651（−26%） | 711（−19%） |
| 波音リツ anywhere | 1282 Hz | 1021（−20%） | 1197（**−7%**） |
| 波音リツ ARROW | 775 Hz | 745（−4%） | 856（**+10%**。行き過ぎ） |
| 持ち込み音源（音量合わせ後） | 1499 Hz | 1186（−24%） | 1357（**−9%**） |
| **平均** | — | **約 −17%** | **約 −5%** |

**これは診断として重要です。** よく学習された rectified flow なら 1 step と多 step の結果は
ほぼ一致するはず（それが rectification の目的）で、**差が出ること自体が 1 step 写像が
まだ真っ直ぐになっていない証拠**です。学習継続（「M3 の後の分岐」の①）の根拠が強まりました。

**注意:** 16 step は flow の評価回数が 16 倍で、このリポジトリの売りである 1-step とは別の設定です。
**ここで測った「モデルの高域不足」は推論設定に依存します。** 以前この節に書いていた
−4〜−26% は 1 step 時の値でした。比較するときは step 数を必ず併記してください。
また ARROW の +10% のように**行き過ぎる**こともあり、多 step が常に良いわけではありません。

**8–16 kHz だけは推論設定で戻りません**（持ち込み音源で 0.9% 対 上限 2.9%）。最高域の再現は
学習か GAN の領域です。

**つぶれ（変化量の減少）は軽度でした。** GT 比で時間方向 std 0.92〜0.96、周波数方向 0.91〜0.94。
聴感の「つぶれ」の主因は変化量ではなく**帯域の偏り**（低域に山が寄り、1–2 kHz が上限の半分以下）です。

**話者条件は強く効いています。** `spk_id` を変えると mel が 0.82〜1.20 ln-mel 動きます。
配線や条件付けの問題ではありません。

**ボコーダーは無罪です**（入力の 2–4 kHz を 4.1% → 3.9% とほぼ保つ）。mel を直接見ると、
**bin 80 以上（約 3 kHz 超）が −0.15〜−0.19 ln-mel 不足**し、2–3 kHz が +0.16 過剰でした。
全体の音量は合っています（mean 差 0.04）。`laplacian_var_ratio` は 0.869 で、
**1.0 を下回る＝やや潰れている**方向です。原因は学習不足（eval loss は 30,000 step でも
下降中）と mel の過平滑で、後者には GAN（[条件付きトラック C](#c-gan-の導入)）が設計上の対処です。

#### この過程で見つかった欠陥・修正 4 件

1. **phrase 名が衝突して cache を黙って上書きしていた。** GTSinger の深い構造で
   `song_name()` が親ディレクトリ（`Control_Group`）を返し、連番がファイルごとに 0 へ戻るため、
   1 歌手 1,922 ファイルが **3 つの名前に潰れて**いました。例外にもなりません。
2. **日本語の曲名がすべて消えていた。** 名前を ASCII に削っていたため、日本語題の曲が
   1,922 中 1,723 件で同じ名前になり、曲単位 split が効きませんでした。
3. **`eval_items` は話者ごとの本数。** 3 話者なら 9 サンプルですが 23 話者では 69 になり、
   1 回の eval が mel 図と音声を 138 本書き出して **10 分以上**（単一コア 100%）かかりました。
   多話者では `eval_items: 1` にします。
4. **GTSinger を丸ごと落とすと HTTP 429 で 3 時間半。** 使う wav だけ選んで 9 分にしました。

### 規模の目安

**見積もり:** 20〜50 人 / 合計 100〜300 時間が現実的な最初の本学習案です。ただしこれは実測前の計画値であり、より小さい corpus で先に recipe を確定させます。

**確認済み（実測後の補正）:** 今回は **23 話者 / 約 18 時間**（1 歌手あたり 0.75 時間）で
recipe が成立しました。**eval loss は 30,000 step でもまだ下がり続けており、
更新回数にも素材量にも伸びしろがあります。** 総時間より話者数を優先する構成は機能しています。

### 注意

**確認済み:** `train.py` は gradient accumulation を実装していません。config に `accum_steps` があっても無視されます。実効 batch を増やす計画を立てる場合、accumulation の実装とテストが先に必要です。

**決定:** この段階では GAN を無効のままにします（`gan.enabled: false`）。flow + 再構成が成立し、artifact を分類できる状態になってから、別 run で GAN を導入します（条件付きトラック C）。

### M3 の後の分岐（要ユーザー判断）

**確認済み:** M3 の base には**高域不足**が残っています（進捗節）。M4 へそのまま進むか、
先に base を改善するかは**素材でも設計でもなく、どこに時間と費用をかけるかの判断**なので、
利用者が決めます。3 択とその根拠を並べます。

| 選択 | 根拠 | 見積もり |
|---|---|---|
| **① 学習を続ける**（実施済み） | eval loss が 30,000 step でも**単調減少中**。さらに **flow の step 数を増やすと明るさが戻る**（平均 −17% → −5%）＝**1 step 写像が未収束**という直接の証拠がある | **実費 $2.148 / 約 2 時間 50 分**（下の実施結果）。当初「約 1 時間・$0.3」と見積もったが、**素材の再生成 69 分と通信課金 $0.87 を数えていなかった** |
| **② GAN を別 run で入れる**（トラック C） | `laplacian_var_ratio` 0.869（1.0 未満＝潰れ気味）。過平滑への**設計上の対処**で、起動条件は満たした | base と同じ budget の A/B が要る。①より高く、不安定化の risk もある |
| **③ M4 へ進む** | 本来の次工程。target 固有の細部が乗るので**高域不足も一緒に軽くなり得る** | M4 の見積もりどおり。base の欠点を持ち越す risk はある |

**①は実施済みです（下の結果）。** 仮説どおり 1 step 写像は真っ直ぐになりましたが、
**clip ごとのばらつきが大きく、3 本中 1 本は悪化しました**。

**次にやるべきは②③のどちらでもなく、測り直しです。** n=3 では「さらに継続すべきか」を
判断できません。**clip 数を増やした再測定は手元の CPU でできます**（インスタンス不要・費用ゼロ）。
[`tools/m3_verify.py`](../tools/m3_verify.py) と
[`tools/audio_metrics.py`](../tools/audio_metrics.py) がそのための道具です。

- 30,000 step 版と 60,000 step 版の checkpoint は**両方とも手元にあります**（`.m0data/m3c/`、
  `.m0data/m3/`）。同じ clip 集合で 1 step と 16 step を測れば、追加の課金なしで比較できます。
- **10〜20 clip まで増やす**と、BC+-6 のような悪化が外れ値なのか傾向なのかが分かります。
- その結果を見てから「さらに継続」「②GAN」「③M4 へ進む」を判断します。**①②③のどれを選んでも、進む前に
[`tools/m3_verify.py`](../tools/m3_verify.py) の spectral centroid を base と比較すること**
（内容指標は高域の欠落を検知しません）。

**やってはいけないこと:** 高域不足を「そういうもの」として素通りし、M5 の品質比較まで持ち越す。
比較の時点で原因の切り分けができなくなります。

### ①を実施した結果（2026-08-31）: 仮説は確認された

**確認済み:** 30,000 → **60,000 step** まで継続しました（RTX 4090、**11.74 step/s**）。
成果物は `.m0data/m3c/`（`ckpt_060000.pt` と実験記録）です。

| 成功条件 | 結果 |
|---|---|
| ① 60,000 step 完走 | **達成**。`[resume] ckpt_030000.pt -> step 30000` から継続 |
| ② eval/loss が 0.02311 より下がる | **達成**。**0.01459（−37%）**。40k 0.01637 → 50k 0.01507 → 60k 0.01459 と**まだ下降中** |
| ③ 1 step と 16 step の差が縮む | **達成**。**10.8 → 6.0 点** |

同一クリップで測り直した比較（3 本）:

| | 1 step | 16 step | 乖離 |
|---|---:|---:|---:|
| 30,000 step | −18.1% | −7.3% | **10.8 点** |
| **60,000 step** | **−13.7%** | −7.7% | **6.0 点** |

**1 step が多 step へ寄った**（−18.1% → −13.7%）一方で 16 step はほぼ動いていません。これは
**1 step 写像が真っ直ぐになった**という仮説どおりの動き方です。eval/flow も 0.00620 → 0.00413、
eval/recon は 0.03383 → 0.02093 に改善しました。

**ただし clip ごとにばらつきます（n=3）。**

| clip | 1 step 30k → 60k | 16 step 30k → 60k |
|---|---|---|
| 1st_color | −25.5% → **−9.0%** | −17.0% → **−2.1%** |
| ARROW | −4.4% → −6.0% | +6.3% → **+0.4%** |
| BC+-6 | −24.2% → −26.2% | −11.1% → **−21.6%** |

1st_color は大きく改善、ARROW は行き過ぎが減り、**BC+-6 は悪化**しました。
平均は改善していますが、**一律に良くなったわけではありません**。

**次の判断:** eval loss はまだ下降中なので、さらに継続する余地があります。ただし
**BC+-6 のような悪化例があるため、step 数を増やせば単調に良くなるとは言えません**。
**次の課金の前に、手元の CPU で clip 数を増やして測り直します**（上の「次にやるべきは
測り直しです」）。checkpoint は 30,000 / 60,000 の両方が手元にあります。

**→ 測り直しました。下の節で、この判断も「高域不足＝学習不足」という見立ても差し替わります。**

### 測り直した結果（2026-08-31）: 原因は学習不足ではなく入力の F0 だった

**確認済み:** n=3 を **40 clip（VocalSet 20 歌手）** へ広げ、30,000 / 60,000 step ×
1 / 16 step の 4 通りを手元の CPU で測り直しました（**38 分・費用ゼロ**）。成果物は
`.m0data/m3re/` の `m3_report.json` です。

**まず測り方に誤りがありました。** centroid 上限比の**符号つき平均**を見ていましたが、
clip の半分は上限より明るく（最大 +33%）半分は暗い（最大 −70%）ので打ち消し合います。
**ceiling からの距離（絶対値）**で測り直すと像が変わります。

| 上限比の絶対値の平均 | 1 step | 16 step | 乖離 |
|---|---:|---:|---:|
| 30,000 step | 37.7 | 41.1 | 3.4 点 |
| **60,000 step** | **34.3** | **34.0** | **0.2 点** |

**①の根拠は消えました。** ①を選んだ理由は「多 step にすると明るさが戻る＝1 step 写像が
未収束」でした。60,000 step では 1 step と 16 step が実質同等（**0.2 点差**）で、
**これ以上 step を回しても 1 step 品質は改善しません**。なお 30,000 step 時点の
「多 step で明るさが戻る」という観察自体も符号つき平均による見かけで、絶対値では
16 step のほうが悪い（明るい clip で行き過ぎる）ものでした。

**偏差には構造がありました**（60,000 step / 1 step / target = 波音リツ）。

| 群 | n | 上限比 |
|---|---:|---:|
| 歌唱・女性 | 9 | **+16.1%** |
| 歌唱・男性 | 11 | **−38.1%** |
| 話し声・女性 | 9 | −29.9% |
| 話し声・男性 | 11 | −46.1% |

男女の歌唱は **source の明るさが同じ**（884 Hz 対 889 Hz、上限も 895 対 922）なのに、
**同じ target** へ変換すると 540 Hz 対 1067 Hz になります。target を男性
（JA_Tenor_1）へ替えても男性 source は 535 Hz で動きません。**target 側の問題ではありません。**

**確認済み: 原因は条件として与える F0 です。** `transpose_f0` で content と loudness を
固定したまま F0 だけを動かすクロスオーバーを行いました（同じ 40 clip、target = 波音リツ）。
**上限（GT mel の再合成）にも同じ移調 F0 を渡す**ので、音高が下がれば centroid も下がる
という当たり前の効果は比から落ちます。

| 群 | −12 半音 | 0 | +12 半音 |
|---|---:|---:|---:|
| 歌唱・男性（変換後の絶対値） | 250 Hz | 540 Hz | **1196 Hz** |
| 歌唱・女性（変換後の絶対値） | 378 Hz | 1067 Hz | 1140 Hz |

**男性 source を 1 オクターブ上げるだけで、女性 source をそのまま変換した明るさを超えます。**
逆に女性を 1 オクターブ下げると 378 Hz まで落ちます。出力のスペクトル傾斜は source の
性別でも target でもなく、**条件の F0 に従います**。

**仮説（未検証）:** rectified flow の出発点 `x0` は F0 から作る擬似 mel です。F0 が低いと
倍音が密になり傾きが変わるため、モデルが励起の傾きを出力へ写している可能性があります。
学習データ側の自然な相関（高い音ほど明るい）も効いているはずです。いずれにせよ結合が
強すぎて、**男性の音高を女性 target の音色で鳴らせていません**（波音リツは C4–C5 中心で、
130 Hz を歌った素材が存在しません）。

**話し声には別の要因が残ります。** +12 にしても女性の話し声は −28.2%（0 のとき −29.9%）と
ほとんど動きません。学習素材が歌唱のみなので分布外であり、移調では直りません。

**分岐の結論:**

| | 判定 |
|---|---|
| ① 継続学習 | **不採用。** 根拠（1 step 未収束）が消えた |
| ② GAN | **不採用（今は）。** 過平滑への対処であって、F0 依存の傾斜は直さない |
| **③ M4 へ進む** | **採用。** ただし下の 2 条件を付ける |

1. **変換時の移調を運用規則にする。** male source → female target は **+7〜+12 半音**。
   `tools/svc_convert.py` と `tools/m3_verify.py` の `--transpose` です。
2. **base の pitch augmentation を検討対象として残す。** 話者ごとの音域を広げれば結合が
   緩みますが、SVC では**特徴抽出の前**に行う必要があります（online は `train.py` が拒否）。

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

M3 完了。**base の状態を引き継ぐ点に注意します。**

- 入力は **`ckpt_060000.pt`**（23 話者 / 60,000 step / 141 MB、手元は `.m0data/m3c/`）。
  **別の `run_name` で fine-tune し、base を上書きしないこと**（M3 ゴール 2）。target は
  **spk_id 22**（波音リツ）。構造を変えないので `--init_from ... --finetune` が使えます。
- **base は「暗い」のではなく、出力の傾斜が入力の F0 に強く従います**（M3 測り直し節）。
  fine-tune 前後で [`tools/m3_verify.py`](../tools/m3_verify.py) の spectral centroid を
  **移調あり・なしの両方**で比較し、改善したのか持ち越したのかを必ず記録します。M4 の
  similarity 指標だけを見ていると、この軸は素通りします。
- **male source → female target の変換には `--transpose` を使うこと**（+7〜+12 半音）。
  移調なしの数値だけで M4 の成否を判定しないこと。
- **推論時に入力の音量を触らないこと。** 学習は生の音量で特徴を取ります。ここを揃えないと
  loudness 条件が学習分布からずれ、出力の高域が落ちます（M3 で実測。実質 19 dB の増幅になっていた）。

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

**成果物:** 部分ロード対応の loader、load report、warm-start あり/なしの checkpoint 2 本と比較結果。

**前提:** M2 完了。**未実装:** 現在の `--init_from --finetune` は同一構造の checkpoint を前提としており、SVS→SVC の部分ロードには使えません。

**実施条件:** M3 の学習コストが問題になった場合のみ。再利用できるのは flow backbone・励起・損失・discriminator・NHVSing interface で、phoneme embedding・length regulation・phoneme encoder の入力 projection は再利用できません。

### (B) NHVSing target fine-tune

**目的:** vocoder 側に起因する artifact を切り分けて解消する。

**ゴール:** ground-truth mel を入力しても同じ artifact が出ることを示したうえで、fine-tune 前後を比較した結果。

**実施条件:** 次の 3 つがすべて成立したときのみ実施します。

1. target singer の高音や裏声で vocoder artifact が系統的に出る。
2. ground-truth mel を入力しても同じ artifact が出る。
3. 音響モデルの誤差と vocoder の誤差を切り分けられている。

**現状（2026-08-30）: 起動条件は成立していません。** M0 ゴール 4 の実測で、base 候補の
GTSinger は NHVSing 既知コーパスと同水準の再合成忠実度を示しました（[台帳 7b
節](svc-dataset-ledger.md#7b-nhvsing-にとって-in-distribution-かの実測2026-08-30)）。
M2 でも ground-truth mel 経由と予測 mel 経由がほぼ同等でした。**既存重みをそのまま使います。**

**成果物:** fine-tune 前後の vocoder 重み、同一 mel を通した比較サンプル、artifact の切り分け記録。

**前提:** M4 完了。まず既存重みを固定した baseline を作ってから比較します。**注意:** mel 分布への過適合、データ規約、重み配布条件を別途確認します。NHVSing V3 の配布重みは非商用データセットで学習されています。

### (C) GAN の導入

**目的:** flow 損失と再構成損失だけで学習した mel は平滑になりがちで、子音や息の質感が鈍ります。GAN はこれを鮮明化する後段です。一方で学習を不安定にし得るため、土台が固まる前に入れると「品質が悪いのは音響モデルか GAN か」を切り分けられなくなります。**土台の完成後に、別 run で**入れるのが目的です。

**ゴール:** 次の 3 つが揃った状態を完了とします。

1. GAN なし baseline と**同一の split・seed・更新 budget**で A/B 比較してある。
2. 客観指標と失敗分類の両方で改善が示せている（片方だけの改善は採用理由にしない）。
3. 悪化した run も残してある。GAN は不安定化させ得るので、失敗の条件自体が知見になります。

**成果物:** GAN あり/なしの checkpoint 2 本、A/B の指標と試聴結果、失敗分類。

**前提:** flow + 再構成が安定し、artifact を分類できていること。

**確認済み（2026-08-30、M3 の後）: 起動の根拠は揃いました。**

| 起動条件 | 状態 |
|---|---|
| flow + 再構成が安定している | 済。30,000 step 完走、eval loss 単調減少 |
| **artifact を分類できている** | 済。**高域不足**として特定（bin 80 以上が −0.15〜−0.19 ln-mel、centroid が上限比 −4〜−26%）。ボコーダー由来でないことも切り分け済み |
| 過平滑の兆候 | `laplacian_var_ratio` **0.869**（1.0 未満＝潰れ気味） |

**ただし「M4 完了後」という当初の前提より前倒しになります。** 高域不足が M4 の
fine-tune で軽くなる可能性があるため、**先に学習の継続（M3「M3 の後の分岐」の①）を試し、
それでも残るなら GAN**、という順序を推奨します。GAN を先に入れると「学習不足だったのか
GAN が効いたのか」を切り分けられません。

**実施条件:** 上記が成立してから、**別 run** で導入します。既存 SVS と同様に `gan_start_step` 以降の二段構成とします。baseline checkpoint は上書きしません。

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
