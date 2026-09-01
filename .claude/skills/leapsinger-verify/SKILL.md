---
name: leapsinger-verify
description: 依存やバージョンを変えた後、環境を移した後（Windows ⇄ Linux / vast.ai インスタンス作成後）、学習を始める前に、リポジトリ全経路が壊れていないかを 1 コマンドで確かめる。torch / librosa / Python / CUDA を更新したとき、uv sync や uv add の後、新しいマシンで最初に何かを回すときは必ずこれを使う。
---

# 疎通確認（全経路スモーク）

## いつ使うか

- `uv add` / `uv sync` / `uv lock` を実行した後
- Python・torch・librosa・CUDA・ドライバのいずれかが変わった後
- 新しいマシン（vast.ai インスタンスなど）で最初に作業するとき
- 学習を長時間回す前（壊れた環境で 1 時間走らせるより 3 分測る）

## やること

```bash
uv run python tools/smoke/run_smoke.py
```

これだけ。終了コードが失敗ステージ数なので、**0 以外なら先へ進まない**。所要は GPU で約 3 分。

**smoke は評価ツールを踏みません。** 話者類似度・CER・信号品質・推論 RTF
（`tools/speaker_*.py` / `asr_cer.py` / `signal_quality.py` / `rtf.py`）は
**`uv sync --extra eval` を入れ忘れていても smoke は通ります**。評価を回す前に
別途これを確認すること:

```bash
# **--extra は「これだけにする」指定。** 足す指定ではないので、必要なものを毎回すべて並べる
# （eval だけを渡すと train / export / dev が消えます。実際に踏みました）
uv sync --extra train --extra export --extra dev --extra eval
uv run python -c "import speechbrain, torchaudio; print('eval extra OK')"
```

よく使う変種:

```bash
uv run python tools/smoke/run_smoke.py --device cpu          # GPU なしの環境
uv run python tools/smoke/run_smoke.py --skip preprocess export   # 速く回す
uv run python tools/smoke/run_smoke.py --only libs svc-train      # 一部だけ
uv run python tools/smoke/run_smoke.py --keep                # 失敗調査用に .smoke/ を残す
```

## 何を見ているか

| ステージ | 壊れていたら分かること |
|---|---|
| `libs` | librosa / scipy / onnxruntime / matplotlib / TensorBoard の API 変更 |
| `gen` | mel 計算（librosa 経路）そのもの |
| `svc-train` / `svs-train` | 学習ループ、励起、flow、GAN 判別器 |
| `svc-pp` | SVC 前処理 CLI の 2 段目。shard 契約、**再実行での bit 一致**、実 loader が読めること |
| `svc-resume` | `torch.load` の既定（weights_only）変更と自動再開 |
| `svc-infer` / `svs-infer` | checkpoint 復元、mel 生成、NHVSing ONNX、WAV 出力 |
| `preprocess` | librosa.load、RMVPE、phrase 分割、shard 契約 |
| `export` | torch.onnx.export、onnxsim、fp16 変換、ORT parity |
| `unittest` | 単体テスト全件（`test_*.py` を自動収集。統合テストは既定 skip） |

## 結果の扱い

- **合成波形が入力なので、品質の検証にはならない。** 「配線が壊れていない」ことしか言えない。
  完了レベル 3（実データ）の根拠にしてはいけない。
- PASS したら、必要なら `doc/svc-implementation-status.md` の「実行環境」表を実測値で更新する。
  そのときは [leapsinger-docs](../leapsinger-docs/SKILL.md) の作法に従う。
- FAIL したら `.smoke/` が残る。`--only <失敗したステージ>` で再現してから直す。

## 既知の落とし穴（ここで引っかかったら）

- **`torch.compile` が使えない環境**（Windows、Triton なし、C コンパイラなし）では
  `[excitation] torch.compile unavailable -> harmonic loop fallback` が出るが、これは正常。
  倍音和がループ版に落ちるだけで結果は変わらない（加算順による Δ~1e-6 のみ）。
  Linux では compile が効くはずなので、Linux でこのログが出たら原因を調べる（3〜4 倍遅い）。
- **`preprocess` の初回は RMVPE の重み 181MB をダウンロードする。** ネットワークが無いなら
  `--skip preprocess svs-pp svs-infer`。
- `.smoke/` は `.gitignore` 対象。コミットしない。
- **`ruff check .` は smoke とは別。** lint は「書き方」、smoke は「動くか」を見る。
  設定は `pyproject.toml` の `[tool.ruff]`。既存コードのスタイルと戦う規則
  （E501 / E701 / E702 / E741）は**理由つきで外してある**ので、勝手に有効化しないこと。
- **`--only` で一部だけ走らせるときは、依存する前段も一緒に指定する。** `gen` を飛ばすと
  `svc-train` は「データが無い」で落ちる。これは環境の問題ではない。

## 実際に踏んだ落とし穴（2026-08-30）

**確認済み:** 「unittest ステージだけ見て通ったことにする」と、次を見落とします。
**必ず全ステージの PASS/FAIL を読むこと。**

| 事象 | 原因 | 対処 |
|---|---|---|
| `--device cpu` なのに `CUDA error: devices busy or unavailable` で学習ステージが全滅 | ① `train.py` が device によらず `pin_memory=True`。② torch 2.13 の optimizer が `step()` ごとに `torch.accelerator.current_stream()` を呼び、**CPU tensor しか無くても壊れた CUDA に触る** | ① `_loader_kwargs()` で CUDA のときだけ有効に（回帰テストあり）。② `--device cpu` のとき `CUDA_VISIBLE_DEVICES=-1` を全サブプロセスへ渡す |
| SVC の学習・再開・推論が `model.content_dim=256 but dataset content_dim=768` で落ちる | `configs/svc_base.yaml` を 256 に変えたのに、smoke の合成データが 768 のままだった | `gen_synth_data.py` が **config から読む**ようにした |
| device 自動判定が cuda を選ぶのに、そのあと全部落ちる | 判定が `torch.cuda.is_available()`（driver の有無しか見ない）だった | **実際に `torch.zeros(1, device='cuda')` を確保**して判定する |

**内容指標は音の劣化を検知しません。** content cos / F0 相関 / V/UV が揃って良くても、高域が落ちて「こもった」音になっていることがあります（実測: centroid 620 → 368 Hz で content cos は 0.8217 → 0.8096 しか動かず）。`tools/audio_metrics.py` の spectral centroid と帯域比を必ず併せて見ること。**そして人に聴いてもらうこと** — この不具合は利用者の「音がこもっている」という一言から見つかりました。

**教訓 3 つ。**

1. 合成データを作る側に定数を置くと、config を変えたときに黙って乖離する。**smoke の入力は
   実際の config から導出する。**
2. **`torch.cuda.is_available()` は「使える」を意味しない。** driver が生きていても context 生成が
   失敗する状態は実在する（`nvidia-smi` は正常に見える）。確保して確かめること。
3. **`CUDA_VISIBLE_DEVICES=""` では効かない。`-1` が要る**（空文字は「未設定」として扱われる）。
