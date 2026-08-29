# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリについて

LeapSinger は歌声合成（SVS）用の音響モデルです。ランダムノイズではなく **F0 から作った「擬似 mel」（倍音インパルス＋白色ノイズ）を rectified flow の出発点 `x0`** にすることで、1 ステップ（`num_steps: 1`）で mel を生成します。mel → 波形は別リポジトリの NHVSing ボコーダー（`checkpoints/nhv_v3*.onnx`）が担当します。

現在のブランチ `feature/svc` では、既存 SVS を残したまま **歌声変換（SVC）経路**を追加中です。設計・調査ドキュメントは `doc/svc.md` が索引になっています（作業前に必ず読むこと）。

```text
SVS: 音素 + duration + F0        -> LeapSinger -> mel + F0 -> NHVSing -> WAV
SVC: source WAV -> content/F0/UV/loudness -> LeapSVC -> mel + F0 -> NHVSing -> WAV
```

## コマンド

**Python は `uv` 経由で実行します。**素の `python` / `pip`、および `uv pip` は使わないこと。依存の追加は必ず `uv add`、実行は必ず `uv run` です。

環境構築:

    uv sync --extra train --extra export       # 依存を .venv へ（推論のみなら uv sync）
    uv add <package>                           # 依存を足すときは常に uv add（pyproject にも記録される）
    uv add --optional train <package>          # extra に足すとき（train / export）

- Python は **3.13 固定**です（`.python-version` と `pyproject.toml` の `requires-python = ">=3.13,<3.14"` の両方）。3.13 に狭めたことで解決が 1 本になり、lock は 3.13 用の最新（librosa 1.0.0 / numpy 2.5.2 / scipy 1.18.1 など）に揃います。別バージョンで動かす提案をするときは、まずこの固定を変える必要がある点を確認すること。
- `uv.lock` はコミット対象です。依存を変えたら lock の差分も一緒にコミットすること。
- **PyTorch は CUDA 版を pyproject が指定しています。** PyPI の Windows 版 torch は CPU ビルドなので、`[[tool.uv.index]] pytorch-cu130` + `[tool.uv.sources] torch` で PyTorch 公式 wheel index を明示しています。Windows / Linux は `uv sync` だけで GPU 版が入り、macOS は marker で PyPI（CPU/MPS）に落ちます。CUDA channel を変えるときは index の url（`cu126` / `cu128` / `cu130` / `cu132`）を書き換えて `uv lock` をやり直します。`uv add torch --index ...` を単発で打って pyproject の index 定義と食い違わせないこと。

前処理（SVS のみ。データセット 1 つにつき recipe yaml が必要）:

    uv run python -m preprocess.run --recipe configs/recipes/<db>.yaml   # -> data/<db>/{shard.npz,metadata.json}

学習（SVS も SVC も同じエントリポイント。`model.arch` で分岐）:

    uv run python -m train --config configs/3singer_ritsu3style_uv_gan2d.yaml \
      --data_dirs data/oniku data/natsume data/ritsu \
      --run_name <run> --out_root log --device cuda

    uv run python -m train --config configs/svc_base.yaml \
      --data_dirs data/target --run_name svc_target --out_root log --device cuda

- 同じコマンドを再実行すると `log/<run_name>/ckpt_*.pt` の最新から**自動再開**します。別実験は必ず `--run_name` を変え、base checkpoint を上書きしないこと。
- `--init_from <ckpt> --finetune` は G 重みのみ読み込み、step / optimizer をリセット、D は新規です。**同一構造の checkpoint を前提**としており、SVS → SVC の部分ロードには対応していません（未実装）。

テスト:

    uv run python -m unittest test_svc_model -v
    uv run python -m unittest test_svc_model.HarmonicSVCModelTests.test_forward_and_infer_reuse_flow_with_svc_conditioning

`uv sync` 済みの環境なら `uv run python -m unittest discover` も通ります。ただし top-level の `test_*.py` は `test_svc_model.py` だけなので、収集される 10 件は上と同じです（preprocess / export / 既存 SVS 経路に自動テストはありません）。`uv` を介さず素の Python で走らせると `librosa` 等が無く収集時に失敗するので、その場合は `test_svc_model.py` を直接指定してください。

ONNX / OpenUTAU 書き出し（SVS のみ。実験的）:

    uv run python -m export.cli --ckpt log/<run>/ckpt_050000.pt --out export/<name> \
      --model-name <name> --variant diffsinger --hop 512 --speaker embed

`infer.py` は CLI を持たないライブラリです（`load_acoustic` / `infer_mel` / `infer_svc_mel` / `load_vocoder` / `mel_to_wav`）。手元での再合成は `notebooks/resynth.ipynb`、書き出し確認は `notebooks/export_and_use_onnx.ipynb` を使います。

## アーキテクチャ

### 共有スタック（SVS / SVC 共通）

`HarmonicAcousticBase`（`leapsinger/models/acoustic_base.py`）が条件 `cond` を組み立て、`MelDilatedRectifiedFlow` が mel を直接 rectified flow で精製します。`HarmonicAcousticModel`（`acoustic.py`）が `_excitation_x0`（`harmonic_excitation.py`）と `_recon_loss` を足し、`HarmonicAcousticModelMultiSpk` が低次元話者ベクトル（`spk_bank` → `spk_proj` → cond 加算）を注入します。

損失は **flow 損失 + mel 再構成損失**が土台で、GAN（`gan.d_type: jcu | mel2d`）は `gan_start_step` 以降に後段で入れる二段構成です。GAN 側は `_forward_flow_gan` が `compute_loss` と数式・mask・RNG 順を一致させて再実装しているので、片方だけ変更すると学習が一致しなくなります。

### SVS と SVC の分岐点

分岐は **condition encoder だけ**です。それ以外（励起、flow backbone、損失、GAN、checkpoint 形式、NHVSing 互換 mel）は共有します。

| | SVS | SVC |
|---|---|---|
| 入力 | phoneme + duration + F0 | content + F0 + UV + loudness |
| encoder | `PhonemeEncoder` + `LengthRegulator` | `ContentAdapter`（`modules/encoders/content_adapter.py`） |
| モデル | `HarmonicAcousticModel(MultiSpk)` | `HarmonicSVCModel`（`models/svc.py`、`phoneme_encoder`/`length_regulator` を `None` にする） |
| dataset | `dataset.py` `LeapSingerDataset` / `shard.npz` | `svc_dataset.py` `SVCFeatureDataset` / `svc_shard.npz` |
| config | `model.arch` 未指定 | `model.arch: svc` |
| checkpoint `config.arch` | `harmonic` / `harmonic_multispk` | `harmonic_svc` |

`train.py` は `is_svc = cfg["model"].get("arch") == "svc"` で dataset・collate・eval・forward を切り替え、`infer.py` の `_build_from_config` は checkpoint に保存された `arch` でモデルを再構築します。**SVC を触るときも既存 SVS 経路（preprocess / 辞書 / export 契約）を壊さないこと。**

### config の構造

yaml は `mel` / `model` / `excitation` / `train` / `gan` / `data` の 6 セクションです。`mel` は `MelSpec`（`leapsinger/config.py`）として前処理・loader・励起 hop で共有され、常に一致している必要があります（44.1 kHz / hop 256 / n_fft 2048 / 128 mel / 40–16000 Hz、NHVSing V3 互換）。

`configs/.gitignore` は top-level の `configs/*.yaml` を無視し、`3speaker_gan2d.yaml` / `3singer_ritsu3style_uv_gan2d.yaml` / `svc_base.yaml` の 3 つだけを公開対象にしています。新しい config を追加してもコミット対象にならない点に注意。

### SVC のデータ契約（厳格）

```text
data/<db>/metadata.json     # {"content_dim": 768, "phrases": {"<name>": <frames>}}
data/<db>/svc_shard.npz     # <name>|content [T,C] / |f0_interp [T] / |uv [T] / |loudness [T] / |mel [128,T]
```

**すべての `T` は完全一致させます。** loader（`svc_dataset.py`）は暗黙の transpose や補間を行わず、幅・フレーム数の不一致を例外にします。この「黙って直さない」性質は前処理ミスを早期に露出させるための設計なので、緩めないこと。

**WAV からこの shard を生成する前処理はまだリポジトリ内に存在しません**（ContentVec/HuBERT + RMVPE + loudness の抽出器が未実装）。SVC の学習を回すには外部で shard を用意する必要があります。実装する際は encoder/model revision、層、sample rate、hop、loudness 定義、F0 extractor version を manifest に記録します。

## この開発での約束事

`doc/` 配下の文書は確度ラベル（**確認済み / 決定 / 推奨 / 見積もり / 仮説 / 未実装 / 要ユーザー判断**）で記述を区別しています。ドキュメントを更新するときはこのラベル体系を維持してください。特に `doc/svc-prior-art-license.md` の主張ルールに従います。

- 「確認済み」はコード・実行 artifact・一次資料のいずれかを示せる場合のみ。
- 「Seed-VC より良い」は同一 test set の blind comparison 後にのみ使う。
- 「リアルタイム」は対象ハードウェアでの end-to-end latency 実測と連続動作後にのみ使う。
- 「世界初」「唯一」は使わない（rectified-flow SVC も harmonic modelling も先行研究がある）。
- 「1-step」は acoustic flow の step 数であり、pipeline 全体の話ではない。

現在の到達点は「実装レベル」と「合成 smoke レベル」までです。実データ学習、Seed-VC 比較、streaming student は未到達です（`doc/svc-implementation-status.md` の検証済み / 未検証の境界を参照）。

## 既知の落とし穴

- `train.py` は **gradient accumulation を実装していません**。config に `accum_steps: 2` があっても無視されます（互換のために残されている値）。実効 batch を増やす提案をする際はこの前提を確認すること。
- SVC では online `pitch_aug` を使えません（特徴量が事前計算済みのため）。`train.py` が明示的に SystemExit します。augmentation は特徴量抽出前に行います。
- **Windows では `torch.compile` が使えません。** `harmonic_excitation.py` の倍音和は compile 前提の融合版（Linux + Triton で 3〜4 倍）ですが、Windows には Triton wheel がなく、さらに日本語ロケール（cp932）では inductor の template 読み込み自体が `UnicodeDecodeError` になります。`triton-windows` を入れても C コンパイラが必要です。コードは **compile 生成時と初回呼び出しの両方**でループ版へフォールバックします（数値差は加算順のみ）。最初から切るなら `LEAPSINGER_EXC_COMPILE=0`。
- 1 曲しかない DB は eval split が空になります（`n_hold = min(eval_songs, 曲数 - 1)`）。`train.py` の `log_eval` は空なら評価を飛ばします。数フレーズの overfit 検証ではこの経路を通ります。
- RMVPE はマルチプロセスで動かさないこと（README の注意）。RMVPE の重み `preprocess/algorithms/rmvpe.pt` は初回実行時に HuggingFace から自動ダウンロードされます（約 181 MB、`.gitignore` 対象）。
- `dataset.py` の phrase 名は `{song}_{NNNN}` 形式で、`_song_of()` が曲単位の train/eval 分割に使います。この命名を崩すと leakage 防止が効かなくなります。
- ライセンス境界: コードは MIT ですが、同梱ボコーダー ONNX、Release 配布の学習済みモデル、学習に使った歌声 DB は MIT 対象外です。Seed-VC（GPL-3.0）は外部 baseline として実行するだけで、コードをこのリポジトリへ取り込まないこと。
