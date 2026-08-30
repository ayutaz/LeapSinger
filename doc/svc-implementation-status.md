# SVC 実装状況と再現手順

確認日: 2026-08-30

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
| [`test_svc_preprocess_integration.py`](../test_svc_preprocess_integration.py) | 追加済み | 実モデルを使う統合テスト（既定 skip） |
| [`test_svc_preprocess.py`](../test_svc_preprocess.py) | 追加済み | align / subset / loudness の契約テスト（32 件） |
| [`test_svc_dataset.py`](../test_svc_dataset.py) | 追加済み | audit / coverage / split / report の契約テスト（58 件） |
| [`tools/smoke/`](../tools/smoke/) | 追加済み | 全経路の疎通を 1 コマンドで回す（11 ステージ） |
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
  "content_dim": 768,
  "phrases": {
    "song01_phrase000": 512
  }
}
```

`svc_shard.npz`:

| key | required shape | dtype の想定 |
|---|---:|---|
| `<name>|content` | `[T, 768]` | float32 |
| `<name>|f0_interp` | `[T]` | float32 Hz |
| `<name>|uv` | `[T]` | float32 / bool compatible |
| `<name>|loudness` | `[T]` | float32 |
| `<name>|mel` | `[128, T]` | float32 ln-mel |

loader は feature width、mel bins、全配列の `T` を検証し、暗黙 transpose / interpolation を行いません。

## 3. 初期設定

`configs/svc_base.yaml` の主要値:

| 項目 | 値 |
|---|---:|
| sample rate / hop | 44,100 / 256 |
| mel bins | 128 |
| content dimension | 768 |
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
uv run python -m train --config configs/svc_base.yaml \n  --data_dirs data/target \n  --run_name svc_target \n  --out_root log \n  --device cuda
```

実行前に repository 内にない `svc_shard.npz` を外部で準備する必要があります。base と fine-tune は別 `run_name` を使います。

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

- 自動テスト **141 件**が成功（`test_svc_model` 10 / `test_svc_preprocess` 73 / `test_svc_dataset` 58）。重いモデルもネットワークも使いません。実モデルの統合テストは 4 件で、`LEAPSINGER_INTEGRATION=1` のときだけ走ります。
- コマンド guard の回帰テスト **37 件**（`tools/hooks/test_guard.py`）。
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

### 制限付き

`uv run python -m unittest discover` は上記環境で成功します。ただし top-level の `test_*.py` は
`test_svc_model.py` / `test_svc_preprocess.py` / `test_svc_dataset.py` の 3 本で、
**preprocess / export / 既存 SVS 経路そのものには自動テストがありません**（`tools/smoke/run_smoke.py`
が疎通を確認するだけです）。

### 未検証

- 実 WAV から feature shard を生成すること。
- 実歌唱データで loss が収束すること。
- checkpoint から WAV を生成し target timbre を確認すること。
- multi-singer base pretraining。
- SVS checkpoint の安全な部分 warm-start。
- Seed-VC との客観・主観比較。
- NHVSing target fine-tune。
- causal / limited-lookahead model、distillation、streaming I/O。
- GPU peak memory、throughput、training time。
- end-to-end RTF / latency / 長時間連続動作。

## 6. 次の実装順序

1. ContentVec/HuBERT + RMVPE + loudness の再現可能な extractor。
2. 小さな権利確認済み real-audio fixture と preprocessing smoke。
3. 1〜数 phrase overfit と WAV 出力。
4. SVS -> SVC shared-weight warm-start loader と load-report tests。
5. multi-singer base と target fine-tune。
6. Seed-VC comparison suite と blind review artifact。
7. offline gate 後に streaming student。

## 7. ブランチと作業ツリー

SVC 開発用ブランチ名は、単なる `svc` より目的が明確な `feature/svc` を採用しました。既存の未コミット変更を stash、restore、reset せず、そのまま保持して切り替える方針です。

この文書の状態一覧は記録時点の snapshot です。最新の branch と dirty state は作業開始・終了時に `git status` と branch 名を再確認してください。
