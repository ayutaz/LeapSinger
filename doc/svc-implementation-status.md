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
| [`README.md`](../README.md) / [`README.en.md`](../README.en.md) | 更新済み | 実験的 SVC の入口と未実装境界 |

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

```powershell
uv run python -m train --config configs/svc_base.yaml `
  --data_dirs data/target `
  --run_name svc_target `
  --out_root log `
  --device cuda
```

実行前に repository 内にない `svc_shard.npz` を外部で準備する必要があります。base と fine-tune は別 `run_name` を使います。

## 5. 現在までの検証

### 確認済み

- SVC targeted unit tests 10 件が成功。
- padding された frame が有効 frame に影響しないこと。
- feature width / frame alignment の不正入力を拒否すること。
- speaker-conditioned encoding の shape と分岐。
- SVC condition から flow forward / inference への配線。
- checkpoint から SVC model を再構築できること。
- 単一 item inference と batch collate。
- repository 内 53 Python file の AST parse。
- `configs/svc_base.yaml` の YAML parse。
- synthetic tensor による train/flow wiring smoke。
- その時点の `git diff --check`。

### 制限付き

全体の `unittest discover` は、借用した PyTorch environment に `librosa` がなく collection 時に停止しました。SVC targeted tests の成功と、repository 全 test suite の成功は区別します。

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
