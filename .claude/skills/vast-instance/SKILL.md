---
name: vast-instance
description: vast.ai の Linux GPU インスタンスを借りて学習を回し、成果物を回収して破棄するまでの手順。GPU が必要になったとき、インスタンスを探す・作る・接続する・止めるとき、リモートで学習を走らせるときに使う。時間課金なので、作成と破棄の判断規則を含む。
---

# vast.ai で学習を回す

**手元の Windows 機は開発・推論・検証用。学習は vast.ai の Linux インスタンスで行う。**
時間課金なので、借りている時間が短くなるように順序を決める。

## 0. 準備（初回のみ）

```bash
uv sync --extra ops        # vastai CLI が入る
```

API token は `.env`（`.gitignore` 対象）に置く。`tools/vast.py` が
`VAST_API_KEY` / `VASTAI_API_KEY` / `VAST_AI_API_KEY` / `VAST_TOKEN` / `VASTAI_TOKEN` の
どれかを自動で拾う。**token を表示・コピー・コミットしない。**

## 1. 借りる前に手元で済ませる

インスタンスは起動した瞬間から課金される。次は**借りる前に**終わらせておく。

- コードを `origin/feature/svc` に push しておく（bootstrap が clone する）
- config を決めておく（[leapsinger-experiment](../leapsinger-experiment/SKILL.md) の 1〜2）
- データの転送手段を決めておく（shard をどうやってインスタンスへ置くか）
- 手元で `uv run python tools/smoke/run_smoke.py --device cpu` を通しておく

## 2. 探す（課金なし）

```bash
uv run python tools/vast.py search --vram 24 --max-price 0.60
```

既定の絞り込み: `num_gpus=1` / `reliability > 0.98` / `verified=true` /
`cuda_max_good >= 13.0`（cu130 wheel を使うため）/ `disk_space >= 60` / 安い順。

VRAM の目安は [`doc/svc-data-compute.md`](../../../doc/svc-data-compute.md)。
target fine-tune は 16 GB、multi-singer base は 24 GB が基準。

**ディスクは 40 GB 以上。** torch cu130 + nvidia 系 wheel だけで 8 GB 前後を使う。

## 3. 作る（ここから課金）

```bash
uv run python tools/vast.py create <offer_id> --disk 60 --yes
```

- `--yes` が無ければ実行されず、料金（$/hr、24h、週）の概算だけ出る。**まず --yes 無しで確認する。**
- 既定イメージは `vastai/base-image:cuda-13.0.3-auto`（ホスト CUDA 13.0 系＝cu130 wheel と一致、
  gcc 入りなので Linux では `torch.compile` が効く）。
- `--rmvpe` を付けると前処理用の RMVPE 重み（181 MB）も落とす。
- onstart で `tools/vast_bootstrap.sh` が走る。uv 導入 → clone → `uv sync` → CUDA 疎通 →
  **倍音和の compile 経路が効いているかの確認** → 単体テスト、まで自動。

## 4. 接続して確認する

```bash
uv run python tools/vast.py instances
uv run python tools/vast.py ssh <instance_id>
uv run python tools/vast.py logs <instance_id>
```

インスタンス上で `/root/bootstrap.log` を見る。**次を確認してから学習を始める。**

1. `is_available: True` と GPU 名・VRAM が期待どおり
2. `triton: <version>` が出ている
3. `compiled path active: True` — **False なら励起が 3〜4 倍遅いまま回ることになる**ので、
   理由（gcc が無い等）を潰してから始める
4. `unittest` が OK

不安があれば全経路を回す（3 分）:

```bash
uv run python tools/smoke/run_smoke.py
```

## 5. 学習する

[leapsinger-experiment](../leapsinger-experiment/SKILL.md) に従う。長時間になるので
バックグラウンドで回し、通知を待つ（sleep でポーリングしない）。

## 6. 成果物を回収する（破棄の前に必ず）

**`destroy` するとディスクごと消える。取り消せない。**

回収するもの: `log/<run>/ckpt_*.pt`、`events.out.tfevents.*`、config のコピー、
`uv.lock`、`nvidia-smi` の出力、生成サンプル。

```bash
uv run python tools/vast.py scp-url <instance_id>   # 転送先の URL を得る
```

学習中も定期的に退避する。インスタンスは落ちることがある。

## 7. 破棄する

```bash
uv run python tools/vast.py destroy <instance_id> --yes
```

- **回収が終わったことを確認してから。** `--yes` 無しなら警告だけ出て実行されない。
- 使い終わったら必ず破棄する。止め忘れがそのまま課金になる。
- 破棄した後に `uv run python tools/vast.py instances` で残っていないことを確認する。

## やってはいけないこと

- `vastai create instance` / `vastai destroy instance` を直接叩く（料金確認と `--yes` が飛ぶ。
  hook で止まる）
- token を echo する、ログに残す、コミットする
- 回収前に破棄する
- インスタンスを立てたまま長時間の調べ物をする（手元でできることは手元でやる）
