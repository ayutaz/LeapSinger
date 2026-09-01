# SVC 実装状況と再現手順

確認日: 2026-09-01

対象ブランチ: `feature/svc`

## 1. 現在の実装

| ファイル | 状態 | 役割 |
|---|---|---|
| [`leapsinger/models/svc.py`](../leapsinger/models/svc.py) | 実装済み | `HarmonicSVCModel`、condition、flow、loss、inference |
| [`leapsinger/modules/encoders/content_adapter.py`](../leapsinger/modules/encoders/content_adapter.py) | 実装済み | content + F0 + UV + loudness の frame-wise adapter |
| [`svc_dataset.py`](../svc_dataset.py) | 実装済み | `metadata.json` / `svc_shard.npz` loader と collate |
| [`train.py`](../train.py) | SVC 分岐を追加済み | `model.arch: svc` の model/dataset/evaluation 配線 |
| [`infer.py`](../infer.py) | SVC 推論を追加済み | 単一 item の mel inference |
| [`configs/svc_base.yaml`](../configs/svc_base.yaml) | 追加済み | offline single-target 初期設定 |
| [`test_svc_model.py`](../test_svc_model.py) | 追加済み | SVC model / dataset / wiring の targeted tests |
| [`preprocess/svc/align.py`](../preprocess/svc/align.py) | 実装済み | SSL 50 Hz → mel grid の **left（直前保持）**整列 |
| [`preprocess/svc/subset.py`](../preprocess/svc/subset.py) | 実装済み | ContentVec 768 → 256 次元の部分集合（seed と index を manifest へ） |
| [`preprocess/svc/loudness.py`](../preprocess/svc/loudness.py) | 実装済み | フレーム log-RMS（mel とフレーム数一致）と dataset 統計での正規化 |
| [`preprocess/svc/audit.py`](../preprocess/svc/audit.py) | 実装済み | M0: clipping / sr / 無音 / DC / 帯域の検査と reject reason |
| [`preprocess/svc/coverage.py`](../preprocess/svc/coverage.py) | 実装済み | M0: 音域帯の滞在時間、percentile、ラベル別滞在時間 |
| [`preprocess/svc/split.py`](../preprocess/svc/split.py) | 実装済み | M0: group 単位 split（層化対応） |
| [`preprocess/svc/report.py`](../preprocess/svc/report.py) | 実装済み | M0: 検査 → coverage → split を 1 度に出す |
| [`preprocess/svc/chunk.py`](../preprocess/svc/chunk.py) | 実装済み | M1: 曲を固定長 phrase へ。**無声のみの chunk を除外** |
| [`preprocess/svc/extract.py`](../preprocess/svc/extract.py) | 実装済み | M1 の 1 段目。encoder と F0 抽出器は引数で受け取る |
| [`preprocess/svc/encoders.py`](../preprocess/svc/encoders.py) | 実装済み | ContentVec / RMVPE の薄い adapter。ここだけが重いモデルに触れる |
| [`preprocess/svc/shard.py`](../preprocess/svc/shard.py) | 実装済み | M1 の 2 段目。`svc_shard.npz` を**決定的に**書く。`features_to_item()` |
| [`preprocess/svc/run.py`](../preprocess/svc/run.py) | 実装済み | M1 の CLI。`--from-cache` で 2 段目だけ再実行 |
| [`tools/m2_verify.py`](../tools/m2_verify.py) | 追加済み | M2 の検証（長さ・F0 追従・V/UV・再現性を測る） |
| [`tools/nhv_indist.py`](../tools/nhv_indist.py) | 追加済み | M0 ゴール 4。コーパスが NHVSing にとって in-distribution かを再合成忠実度で測る |
| [`tools/m3_corpus.py`](../tools/m3_corpus.py) | 追加済み | M3 の素材。GTSinger 20 歌手 + 日本語 3 DB を**話者ごとに 1 shard**で用意し、config を書き出す |
| [`tools/m3_verify.py`](../tools/m3_verify.py) | 追加済み | M3 ゴール 3。未知 source の内容保持を content cos で測る（上限・下限つき）＋ 音の明るさ |
| [`tools/svc_convert.py`](../tools/svc_convert.py) | 追加済み | 任意の WAV を学習済みモデルで変換する CLI。`--self-check` でボコーダー由来の劣化を分離、`--match-loudness` で入力を学習分布へ寄せる |
| [`tools/audio_metrics.py`](../tools/audio_metrics.py) | 追加済み | 帯域エネルギー比と spectral centroid。**内容指標が検知しない高域の欠落**を測る |
| [`tools/speaker_similarity.py`](../tools/speaker_similarity.py) | 実装済み | M4 ゴール 2 の話者類似度。上限・下限・回復率。**既定は ECAPA-TDNN**、12 秒未満のクリップは拒否する |
| [`tools/speaker_calibrate.py`](../tools/speaker_calibrate.py) | 追加済み | 話者照合 encoder が**歌声で使えるか**を 3 群の重なりで判定。**合格条件は事前登録**（`MAX_OVERLAP = 0.20`） |
| [`tools/timing_metrics.py`](../tools/timing_metrics.py) | 追加済み | M5 の timing。onset のずれと**対応が付いた割合**。分解能は hop（5.8 ms） |
| [`tools/asr_cer.py`](../tools/asr_cer.py) | 追加済み | M5 の CER。**上限が判別不能なら差を出さない**。`--language` を素材に合わせること |
| [`tools/signal_quality.py`](../tools/signal_quality.py) | 追加済み | M5 の信号品質（torchaudio SQUIM）。**上限との差のみ**。歌声への妥当性は未検証 |
| [`tools/rtf.py`](../tools/rtf.py) | 追加済み | M5 の推論 RTF。**段ごとに分解**し、特徴抽出と vocoder を含む / 除くを併記 |
| [`tools/pitch_metrics.py`](../tools/pitch_metrics.py) | 追加済み | M5 の F0 相関 / 半音差 / V-UV。**変換済み WAV から測る**ので両システムに当たる |
| [`tools/m5_testset.py`](../tools/m5_testset.py) | 追加済み | M5 の test set を決定的に組む。**有声率の下限**と 1 歌手 1 clip を強制 |
| [`tools/svc_batch.py`](../tools/svc_batch.py) | 追加済み | 複数 clip を 1 プロセスで変換。**変換条件の一致を機械検査**する |
| [`tools/blind_test.py`](../tools/blind_test.py) | 追加済み | blind preference の材料作成と集計。system 名を隠し、順序を randomize |
| [`tools/guard_rail.py`](../tools/guard_rail.py) | 追加済み | 事前登録した判定。**guard rail が 1 つでも落ちたら「より良い」と書けない** |
| [`tools/failure_taxonomy.py`](../tools/failure_taxonomy.py) | 追加済み | M5 ゴール 4。failure を除外せず分類して残す |
| [`tools/normalise_outputs.py`](../tools/normalise_outputs.py) | 追加済み | 外部 baseline の出力を測定ツールが読む形へ揃える |
| [`test_svc_metrics.py`](../test_svc_metrics.py) | 追加済み | 上記の契約テスト（131 件）。重いモデルは引数で受け取る |
| [`configs/svc_target_ft.yaml`](../configs/svc_target_ft.yaml) | 追加済み | M4 の recipe。**checkpoint 選択規則を header に明記**（train loss だけで選ばない） |
| [`configs/svc_base_multi.yaml`](../configs/svc_base_multi.yaml) | 追加済み | M3 の recipe。`spk_map` / `n_speakers` は素材から生成 |
| [`test_svc_preprocess_integration.py`](../test_svc_preprocess_integration.py) | 追加済み | 実モデルを使う統合テスト（既定 skip） |
| [`test_svc_preprocess.py`](../test_svc_preprocess.py) | 追加済み | 整列 / 部分集合 / loudness / shard / 抽出 / chunk / 命名 / 分量選択の契約テスト（115 件） |
| [`test_svc_dataset.py`](../test_svc_dataset.py) | 追加済み | audit / coverage / split / report / GTSinger の wav 選択の契約テスト（63 件） |
| [`tools/smoke/`](../tools/smoke/) | 追加済み | 全経路の疎通を 1 コマンドで回す（12 ステージ） |
| [`tools/hooks/`](../tools/hooks/) | 追加済み | 常に誤りのコマンドを実行前に止める guard と回帰テスト |
| [`tools/vast.py`](../tools/vast.py) / [`tools/vast_bootstrap.sh`](../tools/vast_bootstrap.sh) | 追加済み | vast.ai インスタンスの操作と初期化 |
| `.claude/skills/` | 追加済み | 疎通確認 / 学習実験 / 文書更新 / TDD / vast.ai の手順 |
| [`doc/svc-content-encoder.md`](svc-content-encoder.md) | 追加済み | content encoder の候補比較と決定 |
| [`doc/svc-dataset-ledger.md`](svc-dataset-ledger.md) | 追加済み | M0 の台帳（権利・実測・確定した割り当て） |
| [`doc/svc-plan.md`](svc-plan.md) | 追加済み | M0〜M6 の実行計画（目的・ゴール・完了条件） |
| [`CLAUDE.md`](../CLAUDE.md) | 追加済み | コマンド、共有スタック、SVC データ契約、既知の落とし穴 |
| [`pyproject.toml`](../pyproject.toml) / [`uv.lock`](../uv.lock) / `.python-version` | 更新済み | Python 3.13 固定と CUDA 版 torch の解決を lock |
| [`README.md`](../README.md) / [`README.en.md`](../README.en.md) | 更新済み | 実験的 SVC の入口と未実装境界、uv での環境構築 |

既存 phoneme/duration SVS を置換せず、config で architecture を選びます。

## 2. データ契約

```text
data/<db>/metadata.json
data/<db>/svc_shard.npz
```

`metadata.json`:

```json
{
  "content_dim": 256,
  "frame_rate": 172.265625,
  "phrases": {
    "song01_0000": 512
  }
}
```

`content_dim` は **ContentVec 768 次元から固定ランダムに選んだ 256 次元**です（[選定](svc-content-encoder.md)）。
切り出しは前処理が行い、**loader にはさせません**（「黙って直さない」契約を崩すため）。生の 768 次元は
1 段目の cache に残るので、部分集合を変える ablation は 2 段目の再実行だけで回せます。

`svc_shard.npz`:

| key | required shape | dtype の想定 |
|---|---:|---|
| `<name>\|content` | `[T, content_dim]`（既定 256） | float32 |
| `<name>\|f0_interp` | `[T]` | float32 Hz |
| `<name>\|uv` | `[T]` | float32 / bool compatible |
| `<name>\|loudness` | `[T]` | float32 |
| `<name>\|mel` | `[128, T]` | float32 ln-mel |

loader は feature width、mel bins、全配列の `T` を検証し、暗黙 transpose / interpolation を行いません。

## 3. 初期設定

`configs/svc_base.yaml` の主要値:

| 項目 | 値 |
|---|---:|
| sample rate / hop | 44,100 / 256 |
| mel bins | 128 |
| content dimension | 256 |
| hidden / backbone channels | 256 / 256 |
| speakers | 1 |
| UV | enabled |
| flow steps | 1 |
| max updates | 20,000 |
| max batch frames / size | 30,000 / 16 |
| GAN | disabled |

これは品質最適化済み recipe ではなく、offline target-singer baseline の出発点です。

## 4. 学習例

```bash
uv run python -m preprocess.svc.run --wav-dir download/ritsu --out data/target --device cuda
uv run python -m train --config configs/svc_base.yaml \
  --data_dirs data/target \
  --run_name svc_target \
  --out_root log \
  --device cuda
```

1 行目が WAV から shard を作り、2 行目が学習します。base と fine-tune は別 `run_name` を使います。

## 5. 現在までの検証

### 実行環境（確認済み）

以下は `uv sync --extra train --extra export` で再現できる環境での実測です。

| 項目 | 値 |
|---|---|
| Python | 3.13.13（`.python-version` と `requires-python = ">=3.13,<3.14"` の両方で固定） |
| PyTorch | 2.13.0+cu130（`[[tool.uv.index]] pytorch-cu130` 経由。PyPI の Windows wheel は CPU ビルドのため index 指定が必須） |
| CUDA | wheel build 13.0 / driver 596.21（CUDA 13.2 対応） |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER 16 GB、compute capability 8.9、bf16 対応、cuDNN 9.2 |
| 主要依存 | librosa 1.0.0、numpy 2.5.2、scipy 1.18.1、onnxruntime 1.29.0 |
| 依存解決 | `uv.lock`（89 packages）をコミット済み |

`torch.cuda.is_available()` が True で、2048×2048 の matmul が GPU 上で実行できること（5.22 ms/iter）まで確認しています。**これは環境の疎通確認であり、SVC 学習の throughput・peak VRAM の実測ではありません。**

### バージョン更新後の疎通（確認済み・合成音声）

Python 3.13 / torch 2.13 / librosa 1.0 へ更新した後、次を上記環境で実行して通しました。**入力は合成波形であり実歌唱ではないため、完了レベル 3（実データ）には該当しません。**

| 経路 | 結果 |
|---|---|
| SVC 学習 | `configs/svc_base.yaml` 由来の設定で 40 step。flow 0.0179 -> 0.0127、recon 0.0463 -> 0.0269 |
| 自動再開 | 同一 `run_name` の再実行で `ckpt_000020.pt -> step 20` を読み直して継続 |
| checkpoint | `torch.load(weights_only=True)` で読める（torch 2.6+ の既定で問題なし） |
| SVC 推論 | `infer_svc_mel` が CPU / CUDA 両方で mel [128, 206] を生成 |
| ボコーダー | NHVSing `nhv_v3.onnx` / `nhv_v3x.onnx` を onnxruntime 1.29 で読み、WAV を出力 |
| SVS 学習 | 3.85M params、GAN 有効（`d_loss` / `adv` / `fm` が記録される）で 20 step |
| SVS 前処理 | `preprocess.run` が合成 WAV + lab から `shard.npz` を生成（RMVPE の F0 追従を含む） |
| SVS 推論 | 前処理出力で学習した checkpoint から mel -> WAV |
| ONNX 書き出し | fp32 13.9 MB / fp16 8.9 MB、onnxsim 適用、ORT parity 検証が dynamic shape 2 通りで成功 |
| ログ | TensorBoard の scalar / mel 図（matplotlib）/ 音声が書き出される |

この過程で 2 件の不具合を修正しました。

1. **`torch.compile` が Windows で使えない。** 倍音和は compile 融合版を前提にしていますが、Windows には Triton wheel がなく、日本語ロケール（cp932）では inductor の template 読み込みが `UnicodeDecodeError` になります。compile の**生成時**に例外が出るため、既存の「compile 非対応ならループ」という意図が働かず学習が起動しませんでした。生成時と初回呼び出しの両方でフォールバックするようにしています。`triton-windows` を入れると生成は通りますが、今度は C コンパイラが無く初回呼び出しで失敗します（この経路もフォールバックで吸収することを確認）。
2. **1 曲だけの DB で eval split が空になり `log_eval` が落ちる。** `n_hold = min(eval_songs, 曲数 - 1)` なので単一曲では hold-out が作れません。空なら評価を飛ばすようにしました。[実行計画](svc-plan.md) の M2「1〜数 phrase の overfit」はこの経路を通ります。

### 確認済み

- 自動テスト **366 件**が成功（`test_svc_model` 57 / `test_svc_preprocess` 115 / `test_svc_dataset` 63 / `test_svc_metrics` 131）。重いモデルもネットワークも使いません。実モデルの統合テストは 4 件で、`LEAPSINGER_INTEGRATION=1` のときだけ走ります。
- コマンド guard の回帰テスト **51 件**（`tools/hooks/test_guard.py`）。止めすぎ検出のため、通ってほしいケースも同数以上入れています。
- **実音声 5 コーパスへの検査・coverage・split**（M0。下記「M0 の実データ検証」）。
- padding された frame が有効 frame に影響しないこと。
- feature width / frame alignment の不正入力を拒否すること。
- speaker-conditioned encoding の shape と分岐。
- SVC condition から flow forward / inference への配線。
- checkpoint から SVC model を再構築できること。
- 単一 item inference と batch collate。
- repository 内 53 Python file の AST parse。
- `train` / `infer` / `svc_dataset` / `dataset` / `preprocess.run` / `export.cli` が上記環境で実際に import できること（AST parse より強い確認）。
- `configs/svc_base.yaml` の YAML parse。
- synthetic tensor による train/flow wiring smoke。
- その時点の `git diff --check`。

### M0 の実データ検証（2026-08-30）

入手できた 5 コーパスすべてに検査・coverage・split を通しました。詳細と数値は
[データセット台帳](svc-dataset-ledger.md) 4 節・4b 節です。

| コーパス | ファイル | 時間 | sample rate | 除外 |
|---|---:|---:|---|---:|
| 波音リツ（3 音源） | 150 / 75 曲 | 10.41 h | 44,100 | 0 |
| GTSinger 日本語 | 2,433 / 34 曲 | 6.79 h | 48,000 | 1 |
| VocalSet | 3,613 | 8.73 h | 44,100 | 40 |
| 御丹宮くるみ | 56 | 1.42 h | 96,000 | 3 |
| 夏目悠李 | 52 | 1.20 h | 48,000 | 6 |

**この過程で 10 件の欠陥・訂正が出ました**（split の性別偏り、歌手 ID の取り違えによる leakage、
技法ラベルの表記ゆれ、帯域閾値の誤り、FFT の性能、曲名正規化しないことによる leakage、
無音閾値が曲全体に合わないこと、音域に関する推測の誤りなど）。**いずれも合成データでは
出ませんでした。**

### NHVSing にとっての in/out-of-distribution（2026-08-30、M0 ゴール 4）

**確認済み:** ground-truth mel を NHVSing に通した再合成の忠実度を、5 コーパス × 12 clip ×
6 秒で比較しました（[`tools/nhv_indist.py`](../tools/nhv_indist.py)、詳細と限界は
[台帳 7b 節](svc-dataset-ledger.md#7b-nhvsing-にとって-in-distribution-かの実測2026-08-30)）。

| コーパス | NHVSing 学習 | mel L1 平均 | F0 半音 | V/UV |
|---|---|---:|---:|---:|
| 波音リツ | あり | 0.3470 | 0.019 | 0.989 |
| 夏目悠李 | あり | 0.2969 | 0.025 | 0.981 |
| 御丹宮くるみ | あり | 0.3306 | 0.013 | 0.994 |
| VocalSet | あり | 0.3020 | 0.013 | 0.991 |
| **GTSinger 日本語** | **なし** | **0.3444** | 0.017 | 0.990 |

**判断:** GTSinger は既知 4 コーパスの散らばり（0.2969〜0.3470）の内側で、target singer の
波音リツより良い値です。**base を GTSinger にしてよく、NHVSing の追加学習（条件付きトラック B）を
起動する根拠は現時点でありません。** ただしこれは ground-truth mel の再合成であって、
学習後の音響モデルが出す mel の分布ではありません（M4 の後に測り直します）。

### M1 / M2 の実証（2026-08-30）

**確認済み:** WAV から shard を作り、実音声で overfit して WAV を出すところまで通しました。
学習は vast.ai の RTX A4000（$0.098/hr、M2 一式で 27 分・実費およそ $0.04）で行っています。

| 項目 | 実測 |
|---|---|
| M1 抽出 | 波音リツ 1 曲を 3 秒 chunk で 42 秒。89 chunk 中 **39 件を無声で除外**、50 phrase を採用 |
| M1 再現性 | 同じ cache から作り直して **sha256 が bit 一致**。seed を変えると変わる（ablation が回る） |
| M2 学習 | 2 phrase を 3,000 step。**13〜15 step/s**。flow 0.00710 → 0.00235、recon 0.03573 → 0.01889 |
| M2 出力 | `m2_pred.wav` 3.00 秒 / 44.1 kHz / peak 0.950 |
| M2 F0 追従 | 相関 **0.9991**、中央値 **0.012 半音**、p90 0.075 半音、V/UV 一致 0.988 |
| M2 再現性 | 決定的モードで **bit 一致**（max diff 0.0） |
| 切り分け | ground-truth mel 経由の WAV は F0 相関 0.9990 / 0.012 半音。**予測側とほぼ同等** |

**確認済み:** Linux では `compiled path active: True`（`harmonic_wave` 1.6 ms/call）で、
Windows で使えなかった `torch.compile` 経路が効きます。

**この過程で見つかった欠陥 2 件**（[実行計画](svc-plan.md) M2 の進捗節に詳細）:
無声だけの phrase が学習に入っていたこと、CUDA 推論が既定では bit 再現しないこと。
いずれも合成データでは出ません。

### M2 完了確認で見つかった欠陥（2026-08-30）

**確認済み:** M3 に進む前に M0〜M2 のゴールを 1 つずつ突き合わせたところ、**実装側 2 件・
記録側 1 件**の欠陥が出ました。いずれも「テストは通るが実際には壊れている」種類です。

| 欠陥 | 影響 | 対処 |
|---|---|---|
| smoke の合成 shard が `content_dim=768` 固定で、`configs/svc_base.yaml` は 256 | SVC の**学習・自動再開・推論の 3 ステージが黙って落ちていた** | `gen_synth_data.py` が config から読むように |
| `train.py` が device によらず `pin_memory=True` にしていた | CPU 実行で不要な CUDA 依存が残る | `_loader_kwargs()` に切り出し、CUDA のときだけ有効に。回帰テスト 4 件 |
| `--device cpu` でも torch 2.13 の optimizer が `torch.accelerator.current_stream()` を呼ぶ | **CPU 実行が壊れた CUDA に触って落ちる**（GPU の無い環境向けの経路が使えない） | `run_smoke.py` が `--device cpu` のとき `CUDA_VISIBLE_DEVICES=-1` を渡す（`""` では効かない） |
| smoke の device 自動判定が `torch.cuda.is_available()` | **driver が生きていて context 生成が失敗する状態**で cuda を選び、全ステージが落ちる | 実際に `torch.zeros(1, device="cuda")` を確保して判定 |
| `run_smoke.py` に SVC 前処理 CLI のステージが無かった | M1 の成果物が疎通確認に含まれていなかった | `svc-pp` ステージを追加（bit 一致と実 loader の読み込みまで） |

**教訓 3 つ。** **合成データ側に定数を置くと config との乖離に気づけない**こと、
**`torch.cuda.is_available()` は「使える」を意味しない**こと、そして
**smoke の一部ステージだけを見て「通った」と判断しない**ことです（前回は
`unittest` ステージの件数だけを確認していました）。

同時に M1 ゴール 5 の manifest も 2 項目足りていませんでした（**F0 extractor の version**と
**loudness の窓幅・floor・正規化単位**）。RMVPE 重みの sha256 と入手元、loudness の定義を
記録するようにし、どちらも先に失敗するテストを書いてから実装しています。

### M3（multi-singer base pretraining）の実証（2026-08-30）

**確認済み:** vast.ai の RTX 3090（offer $0.136/hr、disk 150 GB 込みで実効 $0.21/hr）で、
**23 話者・25 shard・約 18 時間**の base を 30,000 step 学習しました。詳細は
[実行計画](svc-plan.md) M3 の進捗節です。

| 項目 | 実測 |
|---|---|
| 素材 | GTSinger 全 9 言語 20 歌手（1 歌手 0.75 h）+ 波音リツ 3 音源 + 夏目悠李 + 御丹宮くるみ |
| dataset | train 8,353 phrase / hold-out 888 phrase。`balance_speakers: true` |
| 学習 | **8.14 step/s**、130 examples/s、151,354 frames/s、30,000 step が約 60 分 |
| **peak VRAM** | **1.95 GB**（見積もりの 24 GB は大きく外していた） |
| checkpoint | 134.7 MB（`HarmonicSVCModel` 11.61M params / `spk_bank` (23, 32)） |
| 損失 | train/flow 0.04717 → **0.00551**、eval/loss 0.02932 → **0.02311（単調減少）**、eval/varL 1.200 → 1.064 |
| 未知 source の内容保持 | content cos **0.8217**（上限 0.9434 / 下限 0.0923、**回復率 85.7%**）。学習済み歌手 84.4% と同等 |
| 抽出 | 25 shard を 52.3 分（GPU 使用率 6% で **GPU 律速ではない**）。shard は約 1.6 GB per audio-hour |

**この過程で見つかった欠陥 4 件**（[実行計画](svc-plan.md) M3 の進捗節に詳細）:
phrase 名の衝突による cache の黙った上書き、日本語曲名の消失、`eval_items` が話者ごとの
本数であること、GTSinger を丸ごと落とすと HTTP 429 で 3 時間半かかること。
**いずれも合成データでは出ません。**

### M3 の継続学習（2026-08-31、30,000 -> 60,000 step）

**確認済み:** 「学習を続けると 1 step 写像が真っ直ぐになる」という仮説を検証し、**確認されました**。
詳細は [実行計画](svc-plan.md) M3「①を実施した結果」。

| 項目 | 30,000 step | 60,000 step |
|---|---:|---:|
| eval/loss | 0.02311 | **0.01459（−37%。まだ下降中）** |
| eval/flow | 0.00620 | 0.00413 |
| eval/recon | 0.03383 | 0.02093 |
| 明るさ 1 step | −18.1% | **−13.7%** |
| 明るさ 16 step | −7.3% | −7.7% |
| **1/16 step の乖離** | **10.8 点** | **6.0 点** |

**1 step だけが多 step へ寄った**のが仮説どおりの動きです。ただし clip ごとには
ばらつき、3 本中 1 本（BC+-6）は悪化しました（n=3）。

実測環境は RTX 4090 で **11.74 step/s**（3090 比 1.44 倍）、peak VRAM 2.0 GB、実費 **$2.148**。

### M4（target fine-tune）の実証（2026-08-31）

**確認済み:** `ckpt_060000.pt`（23 話者 base）から波音リツへ 20,000 step の fine-tune を実施しました。
RTX 3090、`configs/svc_target_ft.yaml`、`--init_from ... --finetune`。base は上書きしていません
（別 `run_name`）。成果物は手元の `.m0data/m4/` に sha256 照合済みで回収してあります。
詳細は [実行計画](svc-plan.md) M4 の進捗節。

| 実測 | 値 |
|---|---|
| 素材 | 波音リツ 3 音源 / 3,414 phrase（train 3,213 / eval 191）/ 約 7.6 時間 / hold-out 6 曲 |
| 速度 | **4.06 step/s** / 64.9 examples/s / 89,491 frames/s / 20,000 step で **82 分** |
| **peak VRAM** | **2.08 GB** |
| eval/loss | 0.01202（2,500 step）→ **0.01129**（20,000 step）。単調減少 |

**確認済み: target の再現と未知 source の内容保持は逆向きに動きます。**

| | target 自己再構成（上限比） | 未知 source の content cos |
|---|---:|---:|
| base（60,000） | 94.8% | 0.8599 |
| ft 10,000（**採用**） | 96.8% | 0.8440 |
| ft 20,000 | 98.1% | 0.8359 |

**採用は `ckpt_010000`。** `configs/svc_target_ft.yaml` に**実験前から書いてあった**規則
（未知 source の cos が base から 0.02 を超えて落ちた checkpoint は選ばない）が 15,000 step で
発動し、通過した候補のうち eval/loss が小さいほうを選びました。**train loss だけで選ぶと
20,000 step を選びます**（M4 ゴール 4 が要求していた失敗の形）。

**確認済み: fine-tune では F0 依存の傾斜は直りません。** 未知 source の明るさの偏差は
33.7 → 37.5 と悪化し、+12 半音の条件でも base 30.0 対 ft 30.8 でした。

**確認済み（2026-09-01）: ゴール 2 の話者類似度も測りました。** 話者照合 encoder を
ECAPA-TDNN に替え、12 秒以上のクリップで較正を通してから測定しています。

| checkpoint | 変換 対 target | 回復率（上限 0.7202 / 下限 0.2862） |
|---|---:|---:|
| base（60,000） | 0.4819 | 45.1% |
| **ft 10,000（採用）** | 0.5244 | **54.9%** |
| ft 20,000 | 0.5380 | 58.0% |

**内容保持が落ちるのと逆向きに、話者類似度は上がります。** 未知 source 6 clip（VocalSet、
各 20 秒、男性は +12 半音）を波音リツへ変換した結果です。**1 群 3 clip と少ないので順序だけを
読み、幅を読まないこと。** 詳細と限界は [実行計画](svc-plan.md) M4 の進捗節。

### M5（Seed-VC 比較）の実証（2026-09-01）

**確認済み:** 事前登録した test set（未知 source 20 + hold-out 6）を両システムで変換し、
客観指標を測りました。詳細は [実行計画](svc-plan.md) M5 の進捗節、記録は `out/m5/record.json`。

| 指標 | LeapSVC | Seed-VC |
|---|---:|---:|
| 話者類似度（回復率） | 0.4712（50.3%） | **0.5912（80.2%）** |
| timing 対応率 | 66.2% | 70.8% |
| F0 相関 | **0.9996** | 0.9978 |
| 半音差（移調を除く） | **0.01** | 0.07 |
| V/UV 一致 | **99.2%** | 98.7% |

**判定: 「Seed-VC より良い」とは書けません**（話者類似度で guard rail が落ちた）。
**F0 追従と V/UV は LeapSVC が上**です。

LeapSVC 単独では、**信号品質は自分の上限とほぼ同等**（stoi −0.006）、**CER は上限から
+0.168** 悪化。推論 RTF は GPU で合計 0.464、**うちボコーダーが 0.432**（93%）。

**未実施:** ゴール 3 の blind listening test（評価者が聴く工程）。材料は `out/m5/blind/`。

### 制限付き

top-level の `test_*.py` は `test_svc_model.py` / `test_svc_preprocess.py` /
`test_svc_dataset.py` / `test_svc_metrics.py` の 4 本（統合テストを除く）で、`run_smoke.py` の `unittest` ステージがこれを自動収集します。
`unittest discover` 自体は hook で止めています（収集条件が暗黙で、走った件数が分かりにくいため）。
**SVS の前処理・export・既存 SVS 経路そのものには自動テストがありません**（`tools/smoke/run_smoke.py`
が疎通を確認するだけです）。

### 未検証

- **音質**（内容保持・F0 追従・V/UV に加えて音の明るさも測るようになったが、それでも音質ではない）。
- ~~話者類似度の測定手段~~ — **解決（2026-09-01）。** ECAPA-TDNN・12 秒以上で較正通過
  （[`tools/speaker_calibrate.py`](../tools/speaker_calibrate.py) で再実行できる）。
- **入力の F0 に対する出力傾斜の結合を緩めること。** 2026-08-31 に 40 clip で測り直し、
  高域不足は学習不足ではなく **F0 条件への強い結合**だと分かった（男性 source を +12 半音
  すると 540 → 1196 Hz）。**変換時の移調（`--transpose`）で実用上は回避できる**が、
  モデル側で緩めるには特徴抽出前の pitch augmentation が要る（**未実装**）。
- **話し声（非歌唱）入力への対応**。学習素材が歌唱のみのため分布外で、移調では直らない
  （+12 にしても −28.2%）。**未実装**。
- ~~CER・信号品質・timing・推論 RTF~~ — **測定済み（2026-09-01、M5）。** 各指標の限界は
  [評価計画](svc-evaluation.md) 4 節。`content_cos` は明瞭度の代理であって CER ではありません。
- **主観的な品質差。** blind listening test が未実施なので、**自然さ・こもり・ノイズは
  記録がありません**。客観指標が拾えない軸です。
- **歌声に妥当な信号品質の指標。** SQUIM も DNSMOS も話し声で学習されており、**歌唱への
  妥当性は未検証**です。上限との差としてのみ使えます。
- **CER の日本語歌唱での信頼性。** 上限 0.0 を確認したのは **1 clip** だけです。
- 256 次元部分集合の seed 比較の**反復**（1 度は実施済み。各 1 run では部分集合の差と run のばらつきを分離できない）。**M4 まで反復せずに進みました。**
- SVS checkpoint の安全な部分 warm-start。
- Seed-VC との客観・主観比較。
- NHVSing target fine-tune。
- causal / limited-lookahead model、distillation、streaming I/O。
- end-to-end RTF / latency / 長時間連続動作。
- **`--transpose` を使った変換の主観品質。** 移調で明るさが戻ることは測りましたが、
  移調そのものが自然さに与える影響は聴いて確かめていません。

## 6. 次の実装順序

1. ~~ContentVec/HuBERT + RMVPE + loudness の再現可能な extractor~~ — **完了**（M1）。
2. ~~実 audio での preprocessing smoke~~ — **完了**（M1、波音リツ実音声）。
3. ~~1〜数 phrase overfit と WAV 出力~~ — **完了**（M2）。
4. ~~multi-singer base~~ — **完了**（M3）。
5. ~~target fine-tune~~ — **完了**（M4）。話者類似度まで測定済み。
6. ~~歌声で較正を通る話者照合 encoder への差し替え~~ — **完了**（ECAPA-TDNN、`eval` extra）。
7. ~~M5 の指標のうち道具が無いもの~~ — **完了**（timing / CER / 信号品質 / 推論 RTF）。
8. SVS -> SVC shared-weight warm-start loader と load-report tests（条件付きトラック A。M3 のコストが問題になった場合のみ）。
9. Seed-VC comparison suite と blind review artifact（M5）。
10. offline gate 後に streaming student（M6）。

## 7. ブランチと作業ツリー

SVC 開発用ブランチ名は、単なる `svc` より目的が明確な `feature/svc` を採用しました。既存の未コミット変更を stash、restore、reset せず、そのまま保持して切り替える方針です。

この文書の状態一覧は記録時点の snapshot です。最新の branch と dirty state は作業開始・終了時に `git status` と branch 名を再確認してください。
