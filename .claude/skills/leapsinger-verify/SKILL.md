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
