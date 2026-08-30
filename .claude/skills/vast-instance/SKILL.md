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
`VASTAI` / `VAST` / `VAST_API_KEY` / `VASTAI_API_KEY` / `VAST_AI_API_KEY` / `VAST_TOKEN` /
`VASTAI_TOKEN` のどれかを自動で拾う。**token を表示・コピー・コミットしない。**
見つからないときは `.env` にある**変数名だけ**（値は出さず）を表示して止まる。

## 1. 借りる前に手元で済ませる

インスタンスは起動した瞬間から課金される。次は**借りる前に**終わらせておく。

- コードを `origin/feature/svc` に push しておく（bootstrap が clone する）
- config を決めておく（[leapsinger-experiment](../leapsinger-experiment/SKILL.md) の 1〜2）
- データの転送手段を決めておく（shard をどうやってインスタンスへ置くか）
- 手元で `uv run python tools/smoke/run_smoke.py --device cpu` を通しておく
- **必要になるスクリプトを書き切っておく。** M3 では課金中に実装を書きたくなり、TDD の順序を
  1 度崩した。素材の構造（GTSinger の階層など）は**借りる前に手元のサンプルで確かめられる**。
- **ディスク量を見積もっておく。** ディスクは実効料金に効く（下の 3 節）。M3 の実測は
  音声 11 GB + shard 29 GB + 環境 11 GB = **51 GB**。
- **通信量の単価を見ておく。** `search` の `$/TB` 列。**環境構築だけで 8 GB 前後**落ちるので
  どの run にも必ず載る。実測で offer 間に **13 倍の開き**（0.3〜4.0 $/TB）があった。

## 2. 探す（課金なし）

```bash
uv run python tools/vast.py search --vram 24 --max-price 0.60
```

既定の絞り込み: `num_gpus=1` / `reliability > 0.98` / `verified=true` /
`cuda_max_good >= 13.0`（cu130 wheel を使うため）/ `disk_space >= 60` / 安い順。

VRAM の目安は [`doc/svc-data-compute.md`](../../../doc/svc-data-compute.md)。
**ただしその表は計画値です。** multi-singer base の実測 peak は **1.95 GB** で、
24 GB 級は要りませんでした（3 節）。`max_batch_size` を上げる・GAN を足す・crop を伸ばす
ときだけ大きい VRAM が要ります。**まず実測を見て、無ければ小さめで試す。**

**ディスクは 40 GB 以上。** torch cu130 + nvidia 系 wheel だけで 8 GB 前後を使う。

## 3. 作る（ここから課金）

```bash
uv run python tools/vast.py create <offer_id> --disk 60 --yes
```

- `--yes` が無ければ実行されない。ただし**料金プレビューは当てにならない**（3b 節。`id=` 検索が
  空を返す）。**検索一覧に出ている `$/hr` を価格の根拠にする。**
- 既定イメージは `vastai/base-image:cuda-13.0.3-auto`（ホスト CUDA 13.0 系＝cu130 wheel と一致、
  gcc 入りなので Linux では `torch.compile` が効く）。
- `--rmvpe` を付けると前処理用の RMVPE 重み（181 MB）も落とす。
- onstart で `tools/vast_bootstrap.sh` が走る。uv 導入 → clone → `uv sync` → CUDA 疎通 →
  **倍音和の compile 経路が効いているかの確認** → 単体テスト（3 ファイル全件）、まで自動。
- **`--disk` は料金に直接効く。** M3 で 150 GB を付けたら $0.136/hr の offer が実効 **$0.21/hr**
  になった（+54%）。必要量を 1 節で見積もってから決める。
- **VRAM は控えめでよいことが多い。** multi-singer base の実測 peak は **1.95 GB** で、
  24 GB 級は要らなかった（`max_batch_size` が先に効くため）。VRAM より
  **回線速度（`down`）とディスク**で選ぶほうが効く場面がある。

## 3b. 実運用で分かったこと（2026-08-30、M2 で一通り回した）

**確認済み:** 次はすべて実際に踏んだものです。

| 事象 | 対処 |
|---|---|
| `vastai execute <id> '<cmd>'` は**制限付き**で、任意コマンドは `Invalid command given` (400) | **SSH を使う。** `execute` は当てにしない |
| SSH には**鍵の登録**が要る | `vastai show ssh-keys` で確認。無ければ `vastai create ssh-key`。**アカウントに登録済みの鍵が手元の鍵とは限りません**（実測: 登録は別マシンの鍵で、手元の `~/.ssh/id_ed25519_vast` では `Permission denied (publickey)` になった）。その場合は**インスタンスへ個別に付ける**: `vastai attach ssh <instance_id> "$(cat ~/.ssh/id_ed25519_vast.pub)"`。数秒で有効になる |
| `vastai destroy instance` は確認プロンプトを出し、stdin が無いと `Aborted.` | `-y` が要る。`tools/vast.py` が渡すようにした |
| `search offers 'id=<N>'` が、その offer が実在しても**空を返す** | `create` の料金プレビューは当てにならない。**一覧に出ている価格を見る** |
| offer ID の**回転が速い** | 検索してすぐ作る。数分置くと消える |
| インスタンスの `logs` には bootstrap の出力が出ない | onstart は `/root/bootstrap.log` へ落としてある。SSH で見る |
| アカウントに**自分が作っていないインスタンス**が居ることがある | `label` と `image` で見分ける。**自分のもの以外は触らない** |
| `show instances` が日本語 Windows で `'cp932' codec can't encode character` で落ちる | `tools/vast.py` が出力を UTF-8 で受けてから安全に表示するようにした |
| `create` の応答に **`instance_api_key` が平文で出る** | `tools/vast.py` が伏せるようにした。**ログにもチャットにも残さない** |

### 3c. M3 で追加で踏んだこと（2026-08-30）

| 事象 | 対処 |
|---|---|
| **リモートで `pkill -f "..."` を打つと自分の SSH シェルごと死ぬ** | コマンド文字列自体がパターンに一致する。`pkill -9 -f "python3 -m trai[n]"` のように**角括弧で自己一致を外す**。**hook が止めるようになりました** |
| `ssh ... 'cmd &'` は SSH が channel を閉じないので戻ってこない | `setsid nohup ... < /dev/null > log 2>&1 &` で完全に切り離し、ログを別途 `tail` する |
| 長時間ジョブの進捗を `ssh` で毎回取ると turn を食う | 完了マーカー（`echo "=== 完了 ==="`）を仕込み、`until grep -q ...; do sleep 60; done` を**バックグラウンドで 1 本**回して通知を待つ |
| ダウンロードの進捗バーが**巨大な出力**になる | 取得するときは `grep -aE "^\[...\]"` などで必ず絞る。生の `tail` を投げない |
| **素材はインスタンスと一緒に消える** | M3 の shard 29 GB は `destroy` で消えた。学習を継続するには**素材の再生成（実測 約 65 分）と checkpoint の再アップロードが要る**。「学習だけ 1 時間」で見積もると外す |
| HF の取得速度は回線ではなく**先方の律速**で決まる | 7.4 Gbps の offer でも 7,977 ファイルに 20 分以上かかった（前回は 900 Mbps の offer で 9 分）。回線速度で選んでも縮まないことがある |

**接続:**

```bash
uv run python tools/vast.py instances          # ssh_host / ssh_port を控える
ssh -i ~/.ssh/id_ed25519_vast -p <port> -o BatchMode=yes root@<host>
ssh -i ~/.ssh/id_ed25519_vast -p <port> root@<host> 'bash -s' < local_script.sh
scp -i ~/.ssh/id_ed25519_vast -P <port> root@<host>:/root/LeapSinger/log/... .
```

**データはインスタンス上で作る。** 転送するより速く、M1 が Linux でも動くことの確認になります。
回線が速い offer を選べば、素材のダウンロードは数十秒で終わります。

```bash
# インスタンス上で
git fetch origin feature/svc && git reset --hard origin/feature/svc   # 手元の push を反映
uv run python preprocess/download_scripts/download_ritsu.py --voice kire
uv run python -m preprocess.svc.run --wav-dir download/ritsu --out data/x --device cuda
```

**実測（RTX A4000 / $0.098 per hour）:** M1 抽出が 1 曲 42 秒、SVC の overfit 学習が
**13〜15 step/s**、M2 一式で 27 分・**実費およそ $0.04**。bootstrap では
`compiled path active: True` / `harmonic_wave 1.6 ms/call` になり、Windows で使えなかった
`torch.compile` 経路が効きます。

**実測（RTX 3090 / offer $0.136 + disk 150GB で実効 $0.21 per hour、M3）:**

| 工程 | 実測 |
|---|---|
| GTSinger の取得（使う wav だけ 7,977 本 / 11 GB） | 9 分 |
| 特徴抽出 25 shard（23 話者・約 18 時間） | 52 分。**GPU 使用率 6%＝GPU 律速ではない** |
| base 学習 | 8.14 step/s、30,000 step が約 60 分 |
| peak VRAM | **1.95 GB**（24 GB 級は要らなかった） |

**実測（RTX 4090 / 実効 $0.371 per hour、M3 の継続 run）:** base 学習 **11.74 step/s**（3090 比 **1.44 倍**）、peak VRAM 2.0 GB。素材の再生成に 69.4 分、HF の取得に 23.9 分。
**実費 $2.148**（GPU $1.157 / download $0.868 / storage $0.111 / upload $0.012）。

**ディスク課金は無視できません。** 150 GB を付けたら $0.136/hr の offer が実効 $0.21/hr に
なりました（+54%）。**必要量を見積もってから付けること。** 上の構成なら実測 51 GB です。

**通信課金はもっと見落とします。** M3 の継続 run の実費 **$2.148** の内訳は
GPU $1.157 / **download $0.868** / storage $0.111 / upload $0.012 で、**通信が 40%** でした。
回線が速い offer を選びましたが、HF 側が律速で速度の恩恵は無く、単価だけ高くつきました。
**`search` の `$/TB` 列を見てから選ぶこと。**

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

**M3 で初めてインスタンスを立てるときは、base 学習の前に seed 0 / seed 1 の比較を済ませる。**
手順は [`doc/svc-plan.md`](../../../doc/svc-plan.md) の M3「開始時にやること」。専用インスタンスを
立てずに済ませるための決定なので、base 学習を先に始めないこと。

## 6. 成果物を回収する（破棄の前に必ず）

**`destroy` するとディスクごと消える。取り消せない。**

回収するもの: `log/<run>/ckpt_*.pt`、`events.out.tfevents.*`、config のコピー、
`uv.lock`、`nvidia-smi` の出力、生成サンプル。

```bash
scp -i ~/.ssh/id_ed25519_vast -P <port> -r root@<host>:/root/LeapSinger/log/<run>/. ./out/
```

学習中も定期的に退避する。インスタンスは落ちることがある。

**M2 で実際に回収したもの:** 生成 WAV 2 本（予測と ground-truth mel 経由）、`m2_report.json`、
TensorBoard の events、shard の `manifest.json`。checkpoint は 141 MB あるので、必要なものだけ選ぶ。

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
