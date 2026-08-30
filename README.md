# LeapSinger

**English**: [README.en.md](README.en.md)

LeapSinger は、CPUでも極めて高速かつ、安定した周期性成分の生成を行える歌声合成のためのDiffusion系の音響モデルです。音素・その長さ（持続長）・音高（F0）を受け取り、メルスペクトログラムを生成します。学習には音素タイミングを記した音素ラベルと音声のみを必要とし、言語による依存はありません（ただし、サンプルには全て日本語のデータを用いています）。

![LeapSinger の概要](doc/fig/leapsinger_overview.png)

**▶ デモを聴く： https://wavtechyukky.github.io/LeapSinger/demo/**

このリポジトリは clone してそのまま使う構成です（PyPI では配布していません）。

## LeapSinger とは

多くの拡散モデルは、ランダムなノイズから出発して、何ステップもかけて少しずつmelを描いていきますが、LeapSinger は**F0 から作った「擬似mel」（インパルス波形にホワイトノイズを足したもの）を出発点にして、rectified flow で1ステップでmelを仕上げます。** （綺麗な周期的成分の生成方法の学習に時間をかけることなく、一足飛びに高品質なmelを生成するためLeapSingerと命名）

擬似melはv/uv対応モデルと非対応モデルで異なっており、v/uv対応モデルは意図的に無声区間を作ることができます。

![v/uv あり・なしの擬似mel](doc/fig/pseudo_mel_vuv.png)

非対応モデル（上）は全フレームに倍音を敷きますが、対応モデル（下）は無声フレームで倍音をゲートします（暗い縦帯が無声区間）。

多くの検討の結果、高品質な歌声合成用のニューラルボコーダーはmelの質感に敏感であり、かつ、音響モデルは周期的成分を綺麗に生成することが課題であることが分かりました。LeapSingerは出発点がノイズではなく、すでに音高の形を持った擬似melなので、最も困難な周期的成分の描写の学習を避けることができます。この設計から、次の効果が得られます。

- **高品質な周期的成分** — ノイズの少ない高品質な周期的成分を安定して描画することができます。これは、ニューラルボコーダーによる合成結果の質感や話者の再現性に良い結果をもたらします。
- **速さ** — Reverse stepは1回のみであり、1コアのCPUで生成を行ってもRTFは0.03を切ります。なお、ステップ数を増やすことはできますが、1step以上行うとかえってGround truthから離れていきます。

このほかの機能:

- **多話者対応** — 話者IDで声を切り替えられます。例として、日本語3話者（御丹宮くるみ/夏目悠李/波音リツ）のモデルを配布します。
- **スタイル変換** — 同じ話者で歌い方（スタイル）を切り替えられます。
- **OpenUTAU 対応（実験的機能）** — OpenUTAUに対応した形式でのエクスポートを用意しています。（※すみません、実際に動作するかは未検証です）

## デモ

**https://wavtechyukky.github.io/LeapSinger/demo/**

- 3話者を学習したモデルで、GT（本物の録音）と生成結果を並べて比較しています。なお、単一の話者で学習させるとわずかにmelの話者の再現性が上がりますが、合成品質に大きな差は見られませんでした。
- スタイル変換のデモを載せています。

## 性能

CPUで計測したRTF（Real-Time Factor。小さいほど速く、1未満なら実時間より速い）です。

音響モデル1本あたりのRTFを、Python nativeとONNXでコア数ごとに比較したものです。

| コア数 | Python native | ONNX |
|:--:|--:|--:|
| 1 | 0.027 | 0.090 |
| 2 | 0.026 | 0.063 |
| 4 | 0.027 | 0.058 |
| 8 | 0.026 | 0.054 |
| 10 | 0.024 | 0.065 |

- **Python native** は1ステップなのでコア数にほぼ依存せず、1コアでも RTF 0.027（実時間の約37倍速）です。
- **ONNX** は onnxruntime のオーバヘッドで native より数倍遅くなりますが、それでも実時間の10倍以上の速さです。コア数は4〜8が最速で、全コア（10）ではかえって遅くなります。
- **NHVSing ボコーダー** は CPU で RTF 0.1 未満です（詳細は NHVSing のリポジトリを参照）。

（計測条件：Apple Silicon 10コア・onnxruntime CPU・約7秒のフレーズ・中央値。機種によって変わります。）

フレーム設定は44.1kHz・hop size256です。hop size512 には、隣り合う2フレームの平均を取ることで対応します。

## 構成

上の図の流れは、次のとおりです。

1. **入力** — 音素・持続長・F0（必要なら話者ID）。
2. **Encoder** — 音素を埋め込み、持続長にしたがってフレーム長へ引き伸ばし、F0 と話者を足してconditionを作ります。
3. **励起（Harmonic + Noise Excitation）** — F0から、インパルス波形にホワイトノイズを足した「擬似mel」を作ります。これがflowの出発点です。
4. **Rectified Flow（1ステップ）** — 擬似メルを出発点に、conditionを条件にして、1ステップで本物らしいmelへ変換します。
5. **NHVSing** — melを音声（波形）に変換する、CPUでRTF0.1を切り、高音質なニューラルボコーダーです。 https://github.com/wavtechyukky/NHVSing/

擬似melは、倍音の減衰・本数・ホワイトノイズの強さを調整できます。しかしながら、検討を重ねた結果、インパルス波形はナイキスト周波数まで重ね切った方が品質に貢献します。

## 使い方

### 環境構築

Python 3.13 固定です（`pyproject.toml` の `requires-python` と `.python-version` の両方で 3.13 に固定）。依存の管理と実行は [uv](https://docs.astral.sh/uv/) に統一しています。

    git clone https://github.com/wavtechyukky/LeapSinger
    cd LeapSinger
    uv sync                               # 推論・再合成・ノートブック
    uv sync --extra train                 # 学習（TensorBoard を追加）
    uv sync --extra export                # ONNX 書き出し
    uv sync --extra train --extra export  # 全部入り

Python の実行は `uv run python ...`、依存の追加は `uv add <package>` です（素の `python` / `pip`、`uv pip` は使いません）。`uv sync` は `.python-version` の Python を自動で用意し、`uv.lock` の解決結果どおりに `.venv/` を作ります。

- **PyTorch** は `pyproject.toml` の `[[tool.uv.index]]` で CUDA 版 wheel（cu130）を指定しているので、Windows / Linux では `uv sync` だけで GPU 版が入ります（PyPI の Windows 版 torch は CPU ビルドのため、この指定が必要です）。別の CUDA を使うときは url の `cu130` を `cu126` / `cu128` / `cu132` などに差し替えて `uv lock` をやり直してください。macOS は marker で PyPI の CPU/MPS ビルドに落ちます。
- GPU が見えているかは次で確認できます。

      uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
- **F0抽出（RMVPE）** の重みは初回実行時に自動ダウンロードされます（HuggingFace → `preprocess/algorithms/rmvpe.pt`）。
- **ボコーダー（NHVSing）** は `checkpoints/` に ONNX 同梱済みで、追加ダウンロード不要です。
- 学習・配布用の**音響モデル本体は Release で配布**しています（リポには含みません）。

### config設定

学習や書き出しの設定は yaml で行います（`configs/` に例があります）。yaml は `mel` / `model` / `excitation` / `train` / `gan` / `data` の6つのセクションに分かれています。主な項目は次のとおりです。

- `model` — `spk_dim`（0より大きいと多話者）、`n_speakers`、`n_styles`（0より大きいとスタイル機能を使う）、`use_uv`（v/uvを条件に使うか）
- `excitation` — `n_harm`（倍音の本数）、`harm_decay`（倍音の減衰）、`noise_ratio`（ホワイトノイズの強さ）
- `train` — `lr`、`max_updates`、`num_steps`（推論のステップ数。**1** を推奨）、`balance_speakers`（話者を均等にサンプリング）など
- `gan` — 質感を鮮明化する GAN の設定。`enabled`（`false` で flow損失＋mel損失のみの学習）、`gan_start_step`（GAN を入れ始める step）、`gan_strength`（敵対的損失の強さ）など
- `data` — `spk_map` / `style_map`（データセットのフォルダ名 → 話者ID / スタイルID の対応）

### 辞書の作成

日本語の音素の一覧は `dict/ja.phonemes` にあります（1行に1音素、並び順がそのままID、先頭の `pau` が ID 0、`#` から先はコメント）。日本語以外や独自の音素を使いたいときは、同じ形式のファイルを用意し、各コマンドに `--phonemes <ファイル>` で渡します（省略時は日本語の `dict/ja.phonemes` を使います）。

### preprocess

データセット1つにつき、yaml（`configs/recipes/<db>.yaml`）を用意します。1曲あたり、音声 `wav` と音素タイミングの `.lab`（＋譜面）を使います。次のコマンドで `data/<db>/` に前処理済みのデータができます。

    uv run python -m preprocess.run --recipe configs/recipes/<db>.yaml

例に使った3つのデータベースは、次のスクリプトでダウンロードできます（各DBの規約に従ってご利用ください）。

    uv run python preprocess/download_scripts/download_oniku.py
    uv run python preprocess/download_scripts/download_natsume.py
    uv run python preprocess/download_scripts/download_ritsu.py

F0の抽出にはRMVPEを使います（RMVPEはマルチプロセスで動かさないようご注意ください）。

### training

学習は2段構成です。前半は flow損失＋mel損失で土台の声を学習し、後半で GAN を入れて質感を鮮明化します（切り替えは config の `gan` セクションで設定。`gan.enabled: false` なら flow損失＋mel損失のみ）。

    uv run python -m train --config configs/<name>.yaml \
      --data_dirs data/<db> [data/<db2> ...] \
      --run_name <name> --out_root log --device cuda

同じコマンドをもう一度実行すると、途中から自動で再開します。

### export

    uv run python -m export.cli \
      --ckpt log/<run>/ckpt_050000.pt \
      --out export/<name> --model-name <name> \
      --variant diffsinger --hop 512 --speaker embed

話者の指定（bake / embed / なし）など詳しい説明は、下の「ONNX/OpenUTAUへの書き出し」を参照してください。

ノートブックで、書き出しから利用まで一通り試すことができます。必要なモデルは Release からダウンロードして `notebooks/sample_data/` に置いてください（詳しくは同フォルダの `place_model_here.txt`）。

    notebooks/export_and_use_onnx.ipynb

- **Part 1** — 通常のONNXエクスポートを行い、モデル単体で動かします（音素＋持続長＋F0 → mel → 音声）。
- **Part 2** — OpenUTAU用のボイスバンクを書き出し、単体版との違いを説明します。

### ONNX / OpenUTAU への書き出し（実験的機能）

`export/` は、チェックポイントを自己完結した ONNXグラフに変換します。励起と1ステップのflowはグラフに焼き込まれるので、使う側は音素・持続長・F0を渡すだけで済みます。話者の扱いは、次から選べます。

- **焼き込み（bake）** — 1つの声を固定します。グラフがいちばん単純になります（話者入力なし）。
- **埋め込み（embed）** — 話者ベクトルを入力にします。1つのグラフでどの声にも切り替えられます。
- **なし（none）** — 単一話者モデル向けです。話者の入力も焼き込みもしません（話者の概念がないグラフ）。多話者モデルではbakeかembedを使います。

OpenUTAU向けには、ボイスバンク一式（ONNX ＋ 設定 ＋ 辞書 ＋ 話者埋め込み）を書き出します。

### ボコーダー

`checkpoints/` に NHVSing ボコーダーを2つ同梱しています。

- `nhv_v3.onnx` — hop size256のmelとF0を受け取ります。
- `nhv_v3x.onnx` — hop size512のmelとF0を受け取ります。

## オプション

- **辞書** — 任意の音素辞書を指定できます。日本語以外にも対応できる設計です（多言語対応そのものは今後の課題です）。
- **v/uv の扱い** — 有声・無声（v/uv）を条件に使うモードと、隙間を線形補間で埋めた連続F0だけを使うモードを選べます。
- **励起の調整** — 倍音の減衰・本数・ホワイトノイズの強さを変えられます。
- **学習レシピ** — 学習の条件は yaml で指定します（使用するデータセット、話者ID、話者ごとのスタイルやデータなど）。

## 今後の課題

### 実験的な SVC 開発

`feature/svc` ブランチでは、事前計算した content feature・F0・V/UV・音量から
既存の harmonic excitation と rectified flow を駆動する、オフライン歌声変換モデルを
開発しています。既存の音素＋durationによる歌声合成経路は変更しません。

現時点ではモデル、特徴量データ契約、学習・評価経路、素材の検査・集計・分割の道具まで
実装済みです。content encoder は **ContentVec**（768次元 layer12、学習には固定ランダムに
選んだ256次元）、F0 は **RMVPE**、SSL から mel grid への整列は **left（直前保持）**に決めています。
ContentVec を実行して特徴量 shard を作る抽出器も実装済みで、WAV からコマンド 1 本で作れます
（再実行で bit 一致）。実音声で overfit して WAV を出し、さらに **23 話者・約 18 時間の multi-singer base 事前学習**まで
通しました（未知の source singer を入れても内容が崩壊しないことを実測）。
**target singer への fine-tune、音質・話者類似度の評価、リアルタイム student は未実施です。**
要件、設計、データ/GPU、学習、評価、先行研究・ライセンス、実装状況、出典を分割した
調査ドキュメントは [doc/svc.md](doc/svc.md) を索引として参照してください。

- **多言語対応** — 現状は日本語データで検証していますが、設計自体は言語に依存しません。他言語の音素辞書・データへの対応を行い、1つのモデルで複数の言語を扱えることを目標とします。
- **さらなる品質向上** — 擬似melの生成方法やパラメータの調整などで話者再現性に改善の余地があると考えています。
- **OpenUTAU 実機での確認** — OpenUTAU 向けのエクスポートは用意していますが、実機での動作は未検証です。実際に動かして確認します。

## ライセンス

コードは MIT です（`LICENSE`）。ただし、同梱のボコーダー ONNX（`checkpoints/nhv_v3*.onnx`）、および Release で配布する学習済みモデルとその学習に使った歌声データベースは MIT の対象外で、それぞれのライセンス・規約に従います（下の謝辞、およびモデル配布物の `CREDITS.txt` を参照）。

## 謝辞

本モデルの学習に使わせていただいたデータセットと、関連するプロジェクトに感謝します。

- 御丹宮くるみ歌声データベース（御丹宮くるみ） — https://onikuru.info/db-download/
- 夏目悠李（歌声DB制作: アマノケイ／音声提供者: 霧野蒼太） — https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs
- 波音リツ — https://www.canon-voice.com/voicebanks/
- Neural Homomorphic Vocoder — https://www.isca-archive.org/interspeech_2020/liu20_interspeech.html
- dsp（zjlww） — https://github.com/zjlww/dsp

配布する多話者モデルには、各データベースの規約に従って上記のクレジットを表示します。夏目悠李については **歌声DB制作: アマノケイ／音声提供者: 霧野蒼太** を表示し、「夏目悠李の出力音声に関する利用規約」をモデル配布物に同梱します。
