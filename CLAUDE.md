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

前処理（SVS。データセット 1 つにつき recipe yaml が必要）:

    uv run python -m preprocess.run --recipe configs/recipes/<db>.yaml   # -> data/<db>/{shard.npz,metadata.json}

前処理（SVC。WAV ディレクトリから直接。音素ラベルは不要）:

    uv run python -m preprocess.svc.run --wav-dir download/ritsu --out data/ritsu_svc
    uv run python -m preprocess.svc.run --from-cache data/ritsu_svc/_cache --out data/ritsu_svc --subset-seed 1
    # 入れ子の深いコーパス（GTSinger の 1 歌手 = <技法>/<曲>/<Group>/NNNN.wav）
    uv run python -m preprocess.svc.run --wav-dir download/gtsinger/Japanese/JA-Soprano-1 \
      --out data/JA_Soprano_1 --song-parts 1 --max-hours 0.75

**2 段構成です。** 1 段目（重い・GPU）が ContentVec と RMVPE を回して `_cache/` へ、2 段目（軽い・CPU）が整列・正規化・次元削減を行って shard を書きます。`--from-cache` で 2 段目だけを回せるので、**補間方法や 256 次元 seed の ablation に ContentVec と RMVPE の再実行が要りません。**

**`--song-parts` を忘れないこと。** 既定は親ディレクトリ名を曲名にします。`<曲>/<曲>.wav` という配置ならそれで正しいのですが、GTSinger のように深いと**全部の曲が `Control_Group` に潰れます**。曲名は `_song_of()` の曲単位 split に使われるので、潰れると leakage します。`--max-hours` は歌手ごとの分量を曲をまたいで均等に選んで揃えるためのものです（base 事前学習では総時間より話者の多様性が効くため）。

**話者ごとにディレクトリを分けること。** `svc_dataset.py` は speaker id を**ディレクトリ名**から `spk_map` で引くので、1 つの shard に複数話者を混ぜると区別できません。M3 の素材一式は `tools/m3_corpus.py` が用意します。

学習（SVS も SVC も同じエントリポイント。`model.arch` で分岐）:

    uv run python -m train --config configs/3singer_ritsu3style_uv_gan2d.yaml \
      --data_dirs data/oniku data/natsume data/ritsu \
      --run_name <run> --out_root log --device cuda

    uv run python -m train --config configs/svc_base.yaml \
      --data_dirs data/target --run_name svc_target --out_root log --device cuda

    # multi-singer base（M3）。spk_map と n_speakers は素材から生成した config に入る
    uv run python tools/m3_corpus.py --download --out data --write-config log/m3_base/config.yaml
    uv run python -m train --config log/m3_base/config.yaml --data_dirs data/<話者>... \
      --run_name m3_base --out_root log --device cuda

- `train.py` は step 100 / 1,000 / 10,000 で `[perf]` 行を出し、`log/<run>/perf.json` と TensorBoard の `perf/*` に step/s・examples/s・frames/s・peak VRAM を残します。vast.ai は時間課金なので、この値がそのまま料金の見積もりになります。
- 同じコマンドを再実行すると `log/<run_name>/ckpt_*.pt` の最新から**自動再開**します。別実験は必ず `--run_name` を変え、base checkpoint を上書きしないこと。
- `--init_from <ckpt> --finetune` は G 重みのみ読み込み、step / optimizer をリセット、D は新規です。**同一構造の checkpoint を前提**としており、SVS → SVC の部分ロードには対応していません（未実装）。

テスト:

    uv run python tools/smoke/run_smoke.py     # 全経路の疎通（合成音声・GPU で約3分）
    uv run python -m unittest test_svc_model -v
    uv run python -m unittest test_svc_preprocess -v  # SVC 特徴抽出前処理（重いモデル不要）
    uv run python -m unittest test_svc_dataset -v     # M0 の素材検査・split・音域集計
    LEAPSINGER_INTEGRATION=1 uv run python -m unittest test_svc_preprocess_integration -v  # 実モデル（既定は skip）
    uv run python -m unittest test_svc_model.HarmonicSVCModelTests.test_forward_and_infer_reuse_flow_with_svc_conditioning
    uv run python tools/hooks/test_guard.py    # コマンド guard の回帰テスト

`run_smoke.py` は 3rd-party API・学習・自動再開・推論・ボコーダー・前処理・ONNX 書き出しまでを 1 コマンドで通し、終了コードが失敗ステージ数になります。**依存やバージョンを変えた後、環境を移した後、学習を始める前に必ず走らせること。** 入力は合成波形なので品質の検証にはならず、配線が壊れていないことだけを示します。

単体テストは **191 件**（`test_svc_model` 25 / `test_svc_preprocess` 103 / `test_svc_dataset` 63）で、重いモデルもネットワークも使いません。`unittest discover` は hook で止めています（収集条件が暗黙で、走った件数が分かりにくいため）。上の 3 本を明示的に並べるか、`run_smoke.py` の `unittest` ステージを使ってください。後者は top-level の `test_*.py` を自動収集し、件数を表示します。`uv` を介さず素の Python で走らせると `librosa` 等が無く収集時に失敗します。

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

`configs/.gitignore` は top-level の `configs/*.yaml` を無視し、`3speaker_gan2d.yaml` / `3singer_ritsu3style_uv_gan2d.yaml` / `svc_base.yaml` / `svc_base_multi.yaml` の 4 つだけを公開対象にしています。新しい config を追加してもコミット対象にならない点に注意。

### SVC のデータ契約（厳格）

```text
data/<db>/metadata.json     # {"content_dim": 256, "frame_rate": 172.265625, "phrases": {"<name>": <frames>}}
data/<db>/svc_shard.npz     # <name>|content [T,C] / |f0_interp [T] / |uv [T] / |loudness [T] / |mel [128,T]
```

**すべての `T` は完全一致させます。** loader（`svc_dataset.py`）は暗黙の transpose や補間を行わず、幅・フレーム数の不一致を例外にします。この「黙って直さない」性質は前処理ミスを早期に露出させるための設計なので、緩めないこと。

**`content_dim` は 256 です。** `doc/svc-content-encoder.md` の決定により、ContentVec 768 次元から固定ランダムに選んだ **256 次元**を shard に書きます（生の 768 は 1 段目の cache に残るので、部分集合を変える ablation は 2 段目の再実行だけで回せます）。**loader に切り出しをさせません** — 「黙って直さない」契約を崩すためです。`configs/svc_base.yaml` も `content_dim: 256` です。

**WAV から shard を生成する経路は実装済みです。** `preprocess/svc/` の構成:

| モジュール | 役割 |
|---|---|
| `align.py` | SSL 50 Hz -> mel grid（172.265625 Hz）の **left（直前保持）**整列。比が整数にならないので方式が契約になる |
| `subset.py` | ContentVec 768 -> **256 次元**の部分集合（seed 0 が既定。index を manifest へ） |
| `loudness.py` | フレーム log-RMS（**mel とフレーム数が一致**）と dataset 統計での正規化 |
| `audit.py` / `coverage.py` / `split.py` / `report.py` | M0 の素材検査・音域と技法の集計・group 単位 split |
| `chunk.py` | 長い曲を固定長の phrase へ切る。**有声率が `--min-voiced` 未満の chunk は捨てる**（イントロが丸ごと無声になるため。実測では 89 chunk 中 39 件が除外され、うち 35 件は完全に無声だった） |
| `extract.py` / `encoders.py` | 1 段目。ContentVec と RMVPE を**引数で受け取り**、`{content, f0_hz, uv, loudness, mel}` を返す |
| `shard.py` | 2 段目。整列・部分集合・正規化を当てて `svc_shard.npz` を書く。`features_to_item()` は推論側に同じ正規化を当てる |
| `run.py` | CLI。`--from-cache` で 2 段目だけ再実行できる |

**確認済み:** WAV から shard までコマンド 1 本で作れ、再実行で **bit 一致**します（`np.savez` は zip にタイムスタンプを埋めるので自前で決定的に書いています）。

manifest には encoder の model revision と層、sample rate、hop、**SSL の stride と補間方法**、正規化、**256 次元の index と seed**、loudness の定義、F0 extractor version、入力 WAV の checksum を記録します。

## 自動化と安全装置

`.claude/` にこのリポジトリ用の skill と hook を置いています。

| 種類 | 名前 | 役割 |
|---|---|---|
| skill | `leapsinger-tdd` | このリポジトリでの TDD の当て方（重いモデルの扱い、契約テスト、置き場） |
| skill | `leapsinger-verify` | 依存・環境を変えた後の疎通確認の回し方と結果の読み方 |
| skill | `leapsinger-experiment` | 学習実験を事故なく回す手順（run 名、無視される設定、記録、主張の範囲） |
| skill | `leapsinger-docs` | 確度ラベルと主張規則を保ったままドキュメントを更新する作法 |
| skill | `vast-instance` | vast.ai インスタンスの検索・作成・回収・破棄、実運用で踏んだ落とし穴 |
| hook | `tools/hooks/guard_commands.py` | `PreToolUse` で「常に間違い」なコマンドを実行前に止める（回帰テスト 45 件） |

hook が止めるもの: `uv pip` / 素の `pip` / 素の `python`、`.env` の staging、`git push --force`、`git reset --hard`、`log|data|checkpoints|.git` の `rm -rf`、`vastai` の直接叩き（料金確認を飛ばすため）、`unittest discover`、**手元での学習**（device によらず）、**既存 ckpt がある run へ `--init_from` を渡す**こと（`train.py` はこれを黙って無視して自動再開します）、**取得スクリプトの `-m` 実行**（`_gdrive` の兄弟 import が解決できず必ず失敗）、**`CUDA_VISIBLE_DEVICES=""`**（空文字は未設定扱いで CUDA が隠れない。`-1` が要る）。

止めすぎると自動運転が壊れるので、判断の余地がないものだけを対象にしています。どうしても必要なときはコマンド末尾に `# guard:allow` を付けると通ります。ルールを足したら `tools/hooks/test_guard.py` にケースも足してください。

## この開発での約束事

**実装はすべて TDD で行います。** 失敗するテストを先に書き、失敗を確認し、通す最小限のコードを書く。先に書いたテストが無い実装コードは破棄してやり直します。規律は `superpowers:test-driven-development`、このリポジトリ固有の当て方（重い事前学習モデルをどう避けるか、何を契約テストにするか）は `leapsinger-tdd` skill を参照してください。

`doc/` 配下の文書は確度ラベル（**確認済み / 決定 / 推奨 / 見積もり / 仮説 / 未実装 / 要ユーザー判断**）で記述を区別しています。ドキュメントを更新するときはこのラベル体系を維持してください。特に `doc/svc-prior-art-license.md` の主張ルールに従います。

- 「確認済み」はコード・実行 artifact・一次資料のいずれかを示せる場合のみ。
- 「Seed-VC より良い」は同一 test set の blind comparison 後にのみ使う。
- 「リアルタイム」は対象ハードウェアでの end-to-end latency 実測と連続動作後にのみ使う。
- 「世界初」「唯一」は使わない（rectified-flow SVC も harmonic modelling も先行研究がある）。
- 「1-step」は acoustic flow の step 数であり、pipeline 全体の話ではない。

現在の到達点は**完了レベル 3（実データ）**です。実音声から shard を作り、23 話者・約 18 時間の multi-singer base を学習し、未知 source で内容が崩壊しないことまで実測しました（M0〜M3 完了）。**target fine-tune、音質と話者類似度の評価、Seed-VC 比較、streaming student は未到達**です（`doc/svc-implementation-status.md` の検証済み / 未検証の境界を参照）。

## 既知の落とし穴

- `train.py` は **gradient accumulation を実装していません**。config に `accum_steps: 2` があっても無視されます（互換のために残されている値）。実効 batch を増やす提案をする際はこの前提を確認すること。
- **ローカルの GPU は `nvidia-smi` が正常に見えても context 生成に失敗することがあります**（`CUDA error: CUDA-capable device(s) is/are busy or unavailable`）。`torch.cuda.is_available()` は driver の有無しか見ないので **True を返しても使えるとは限りません**。判定するなら `torch.zeros(1, device="cuda")` を実際に確保すること。この状態では推論スクリプトも落ちるので `--device cpu` で回します。
- **CPU で回すときは `CUDA_VISIBLE_DEVICES=-1` を渡すこと。** torch 2.13 の optimizer は `step()` ごとに `torch.accelerator.current_stream()` を呼ぶため、CPU tensor しか無くても壊れた CUDA に触って落ちます。**`""` では効かず `-1` が要ります。** `run_smoke.py --device cpu` は自動で渡します。`train.py` の `pin_memory` も `_loader_kwargs()` で CUDA のときだけ有効です（回帰テストは `test_svc_model.LoaderKwargsTests`）。
- **`tools/smoke/` の合成データは `configs/svc_base.yaml` の `model.content_dim` を読んで作ります。** ここを定数に戻すと、config を変えたときに SVC の学習・再開・推論ステージが黙って落ちます（実際に起きました）。
- **推論時に入力の音量を触らないこと。** 学習（`preprocess.svc.run`）は生の音量のまま特徴を取ります。推論側で peak 正規化すると loudness 条件が学習分布からずれ、モデルが低域を持ち上げて高域を削ります（波音リツ DB は peak 0.107 なので実質 19 dB の増幅になり、spectral centroid が 620 → 368 Hz に落ちました）。`features_to_item()` は正規化の同一性を保証しますが、**その手前で波形を加工すると保証の外**です。
- **内容指標だけで音の劣化を判断しないこと。** 上の不具合で centroid が 620 → 368 Hz に落ちても、content cos は 0.8217 → 0.8096 としか動きませんでした。`tools/audio_metrics.py` の帯域指標を併せて見ます。
- **phrase 名の衝突は例外になりません。** `preprocess.svc.run` は曲ごとの通し番号で採番し、衝突を検出したら止めます。この採番を「ファイルごとに 0 から」に戻すと、同じ曲名の別ファイルが cache を**黙って上書き**し、shard の phrase 数が減るだけになります（GTSinger で 1,922 ファイルが 3 名に潰れました）。
- **曲名を ASCII に削らないこと。** `_SAFE` は `[^\w-]` なので CJK を残します。ASCII だけにすると日本語題の曲がすべて同じ名前になり、曲単位 split が効きません（1,922 中 1,723 件が潰れました）。曲名は casefold して表記ゆれ（`Heartful_Song` と `Heartful_song`）も畳んでいます。
- SVC では online `pitch_aug` を使えません（特徴量が事前計算済みのため）。`train.py` が明示的に SystemExit します。augmentation は特徴量抽出前に行います。
- **学習はすべて vast.ai の Linux インスタンスで行います。手元の Windows で `train.py` を起動すると device によらず hook が止めます**（`tools/hooks/guard_commands.py` の `check_local_training`）。`--device cpu` に逃げるのも不可です。CPU は実測で 1 phrase 1200 step に約 60 分かかり、実験記録の環境も本番と食い違います。ローカル GPU は他の作業と取り合って `unspecified launch failure` を起こしました（実際に発生）。 手元の Windows 機は開発・推論・検証用で、セットアップは `tools/vast_bootstrap.sh`。API token 等は `.env`（gitignore 済み）に置きます。`uv.lock` は Linux も解決済みで、Linux では `triton` が入るため下の `torch.compile` の制約は当てはまりません。
- **Windows では `torch.compile` が使えません。** `harmonic_excitation.py` の倍音和は compile 前提の融合版（Linux + Triton で 3〜4 倍）ですが、Windows には Triton wheel がなく、さらに日本語ロケール（cp932）では inductor の template 読み込み自体が `UnicodeDecodeError` になります。`triton-windows` を入れても C コンパイラが必要です。コードは **compile 生成時と初回呼び出しの両方**でループ版へフォールバックします（数値差は加算順のみ）。最初から切るなら `LEAPSINGER_EXC_COMPILE=0`。
- **`eval_items` は話者ごとの本数です。** 3 話者なら 9 サンプルですが 23 話者では 69 になり、1 回の eval が mel 図と音声を 138 本書き出して 10 分以上（単一コア 100%）かかります。多話者では `eval_items: 1` にして `eval_interval` も大きくすること（実測で踏みました）。
- 1 曲しかない DB は eval split が空になります（`n_hold = min(eval_songs, 曲数 - 1)`）。`train.py` の `log_eval` は空なら評価を飛ばします。数フレーズの overfit 検証ではこの経路を通ります。
- RMVPE はマルチプロセスで動かさないこと（README の注意）。RMVPE の重み `preprocess/algorithms/rmvpe.pt` は初回実行時に HuggingFace から自動ダウンロードされます（約 181 MB、`.gitignore` 対象）。
- `dataset.py` の phrase 名は `{song}_{NNNN}` 形式で、`_song_of()` が曲単位の train/eval 分割に使います。この命名を崩すと leakage 防止が効かなくなります。
- ライセンス境界: コードは MIT ですが、同梱ボコーダー ONNX、Release 配布の学習済みモデル、学習に使った歌声 DB は MIT 対象外です。Seed-VC（GPL-3.0）は外部 baseline として実行するだけで、コードをこのリポジトリへ取り込まないこと。
