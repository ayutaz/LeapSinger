# SVC データセットと計算資源

この文書の時間・GPU・学習所要時間は、実測前の計画値です。モデル品質やメモリ使用量の確認結果として扱わないでください。

## 1. 理想的な音声条件

| 項目 | 推奨 |
|---|---|
| format | mono PCM WAV、16-bit または 24-bit |
| sample rate | 44.1 kHz または 48 kHz。前処理で 44.1 kHz へ統一 |
| source | isolated vocal。元から単独収録が最良 |
| room / effects | reverb、delay、chorus、compression が少ない |
| quality | clipping、dropout、強い定常 noise がない |
| content | 歌唱中心。speech、歓声、重なり声は別管理 |

Demucs 等で分離した stem も候補ですが、伴奏 leakage、位相 artifact、残響が target timbre として学習される危険があります。元の isolated vocal と分離 stem は dataset tag を分け、同じ test set へ混在させません。

## 2. target singer の量

**見積もり:**

| usable singing | 目的・期待 |
|---:|---|
| 30 分〜1 時間 | pipeline の PoC。品質判断には不足 |
| 2〜3 時間 | target fine-tune の初期比較 |
| **5〜10 時間** | 実用品質を狙う最初の推奨範囲 |
| 10〜20 時間 | 音域・発声の安定性を高める |
| 20〜30 時間以上 | 多様な style を同一 target で扱う |

**確認済み（2026-08-30 実測）:** target singer に確定した波音リツは 3 音源合計 **10.41 時間**で、
上表の推奨範囲を満たします。音域は F0 p50 360 Hz、75.3% が C4–C5 です
（[データセット台帳](svc-dataset-ledger.md) 4b 節）。

時間より coverage が重要です。最低限、次を集計します。

- low / mid / high pitch と各 pitch の滞在時間
- chest / head / falsetto、強声 / 弱声、breathy voice
- voiced / unvoiced、子音の種類、速い歌唱
- long tone、vibrato、pitch slide、しゃくり
- 曲、収録日、マイク、部屋、処理 chain

5 時間あっても中音域の同じ歌い方だけなら、高音裏声や強い子音で崩れます。逆に短くても coverage が良いデータは PoC に有効です。

## 3. multi-singer 事前学習量

**見積もり:**

| 合計 singing | 用途 |
|---:|---|
| 10〜30 時間 | multi-singer 配線の PoC |
| 50〜100 時間 | base pretraining の最低推奨候補 |
| 100〜300 時間 / 20〜50 人 | 現実的な最初の本学習案 |
| 200〜500 時間 | より強い汎化を狙う構成 |
| 500〜1,000 時間 | 大規模 base |
| 1,000 時間以上 | 汎用性を広げる大規模研究 |

人数、性別、音域、言語、録音条件の多様性を確保します。ただし diversity のために権利や品質の不明なデータを混ぜません。

**確認済み（2026-08-30 実測）:** 手元に確保できた日本語歌唱は **19.82 時間 / 話者 5 名**です
（波音リツ 10.41 / GTSinger 日本語 6.79 / 御丹宮くるみ 1.42 / 夏目悠李 1.20）。
上表の「100〜300 時間 / 20〜50 人」には**日本語だけでは遠く届きません**。
**決定:** base は GTSinger の全 9 言語（80.59 時間）を軸にし、日本語 3 DB を足す構成にします。
多言語で音素表現を学び、日本語 target へ fine-tune します。

**確認済み: 低音域が全素材で不足しています。** C3（約 131 Hz）より下の滞在時間は最大でも 3.6%
（夏目悠李）で、GTSinger 日本語も `range=low` が 4.8% です。**仮説:** 低い男声を source にした
変換は学習分布の外側への外挿になります。[評価計画](svc-evaluation.md) の failure taxonomy で
観察項目に入れます。

平行データは必須ではありません。各歌手の `content + F0 + loudness -> その歌手の mel` を speaker condition 付きで自己再構成し、話者に依存しにくい変換器を学習します。

## 4. split と leakage 防止

- phrase のランダム分割だけでなく、曲単位で test を分離する。
- 同じ take を切り分けた断片を train と test に跨がせない。
- source separation 前後の同一音源を別 split に置かない。
- target test song は fine-tune、early stopping、threshold 調整に使わない。
- 未知 source singer の test set を別に用意する。
- duplicate / near-duplicate を audio fingerprint 等で検査する。

## 5. 前処理成果物

現在の loader 契約では、dataset directory ごとに以下を置きます。

```text
data/<dataset>/
  metadata.json
  svc_shard.npz
```

`metadata.json` は既存の `phrases` map（phrase name -> mel frames）と `content_dim` を記録します。`svc_shard.npz` には phrase ごとに以下の key を置きます。

| key | shape |
|---|---:|
| `<name>|content` | `[T, content_dim]` |
| `<name>|f0_interp` | `[T]` |
| `<name>|uv` | `[T]` |
| `<name>|loudness` | `[T]` |
| `<name>|mel` | `[128, T]` |

すべての `T` は完全一致させます。loader は silent transpose や自動補間をせず、前処理ミスを明示的に失敗させます。

**確認済み:** WAV からこの shard を再現可能に生成する command は `preprocess.svc.run` として実装済みです。manifest には encoder の id / revision / layer、sample rate、hop、SSL stride、補間方法、loudness の窓幅・floor・正規化単位、F0 extractor の重み checksum と入手元、256 次元部分集合の index と seed、入力 WAV の checksum を保存します。

## 6. GPU メモリの目安

**見積もり:**

| VRAM | 想定 |
|---:|---|
| 8 GB | very small batch / short crop の smoke。調整負担が大きい |
| 12 GB | PoC は可能 |
| **16 GB** | target fine-tune の現実的な基準 |
| **24 GB** | multi-singer base の推奨基準 |
| 32〜48 GB | 長い crop、大きい batch、teacher/student 同時処理に余裕 |
| 80 GB | PoC には不要。大規模化・高速化用 |

この表は計画値です。AMP、gradient checkpointing、feature width、GAN、sequence length、PyTorch/CUDA version で変動します。

**確認済み（2026-08-30 実測、M3 の multi-singer base）: 実際の peak VRAM は 1.95 GB でした。**
上表の「multi-singer base は 24 GB」は**大きく外していました**。23 話者・8,353 phrase・
`max_batch_frames: 30000` / `max_batch_size: 16` / GAN 無効 / fp32 の条件です。

**理由:** phrase が 8 秒（約 1,378 フレーム）なので、30,000 フレームなら 21 phrase 入る計算ですが
`max_batch_size: 16` が先に効いて 1 batch は約 22,000 フレームで頭打ちになります。**VRAM ではなく
batch 本数が制約**でした。24 GB のカードを取っても 2 GB しか使っていません。

**含意:** multi-singer base だけなら **8〜12 GB のカードで足ります**（より安い offer が選べます）。
24 GB 級が要るのは、`max_batch_size` を上げる・GAN を足す・crop を伸ばすときです。
まだ measure していない条件へこの数字を外挿しないでください。

**確認済み:** 現在の開発機は RTX 4070 Ti SUPER 16 GB（driver 596.21 / torch 2.13.0+cu130、詳細は [実装状況](svc-implementation-status.md) の「実行環境」表）です。上表の基準では target fine-tune は手元で回せる一方、multi-singer base pretraining の推奨基準 24 GB には届きません。**決定:** 学習は vast.ai の Linux GPU インスタンスで行い、手元の Windows 機は開発・推論・検証に使います。したがって上表の 24 GB / 48 GB 級はインスタンスの選択で満たせます。**見積もり:** インスタンスは時間課金なので、必要 VRAM だけでなく「1 実験あたり何時間か」で選ぶことになります。最初の 100 / 1,000 update の実測（examples/sec、frames/sec、peak VRAM）が、そのまま料金の見積もりになります。

**確認済み:** `uv.lock` は Linux 側も解決済みで、`torch 2.13.0+cu130` と `triton 3.7.1`、`nvidia-cudnn-cu13` / `nvidia-nccl-cu13` / `cuda-toolkit 13.0.3.0` が入ります。つまり Windows では使えなかった `torch.compile`（倍音和の融合、3〜4 倍）が Linux では有効になります。セットアップは [`tools/vast_bootstrap.sh`](../tools/vast_bootstrap.sh) が行い、compile 経路が実際に効いているかまで確認します。**見積もり:** torch と nvidia 系 wheel だけで 8 GB 前後を消費するため、インスタンスのディスクは 40 GB 以上を見ます（手元 Windows の `.venv` 実測は 3.4 GB、うち torch が 2.8 GB）。

## 7. batch 設計

既存 canonical config の一例は `max_batch_frames: 240000`、`max_batch_size: 128`、`max_updates: 50000` です。一方、SVC 初期設定は保守的に次を使います。

```yaml
max_batch_frames: 30000
max_batch_size: 16
max_updates: 20000
gan.enabled: false
```

**重要:** 既存 config に `accum_steps: 2` があっても、現行 `train.py` は gradient accumulation を実装していません。したがって「30k frames × accumulation 8 = effective 240k」という案は、accumulation 実装・テスト後にのみ有効です。

16 GB GPU の初回探索では `max_batch_frames` 30k から始め、OOM と peak memory を記録しながら 60k まで段階的に上げる案を推奨します。

## 8. 学習時間の目安

**未検証見積もり:** 小規模 target fine-tune の所要時間は、データ長、更新回数、feature cache、GAN の有無で大きく変わります。

| GPU class | 初期 PoC の大まかな期待 |
|---|---|
| RTX 3060 12 GB | 半日〜1 日程度の可能性 |
| RTX 4070 Ti / Super（**現在の開発機がこのクラス**） | 数時間〜半日程度の可能性 |
| RTX 4090 | 数時間程度の可能性 |
| A100 / H100 | PoC には過剰だが大規模 pretraining を短縮可能 |

**確認済み（2026-08-30 実測）:** vast.ai の **RTX A4000 16 GB（$0.098/hr）** で、SVC の overfit
（2 phrase・各 3 秒・content_dim 256・hidden 256）が **13〜15 step/s** でした。M1 の抽出は
1 曲（3 秒 chunk × 50 phrase）で 42 秒です。M2 一式（素材取得・抽出・3,000 step 学習・検証）で
**27 分・実費およそ $0.04** でした。**これは overfit の規模なので、本学習の見積もりには使えません。**

### M3（multi-singer base）の実測（2026-08-30、RTX 3090 24 GB / $0.21 per hour）

**確認済み:** 23 話者・8,353 train phrase（hold-out 888）・約 18 時間の音声での実測です。
料金は disk 150 GB ぶんを含みます（offer 自体は $0.136/hr）。

| 工程 | 実測 |
|---|---|
| GTSinger の取得 | 使う wav だけ選んで **7,977 本 / 11 GB を 9 分**。リポジトリ全体（149,037 ファイル）を落とすと HTTP 429 で律速され **3 時間半** |
| 特徴抽出（2 段） | **25 shard を 52.3 分**。1 歌手 0.75 h ぶんが約 140 秒（**約 19 倍速**）。GPU 使用率は約 6% で、**GPU 律速ではない** |
| shard の容量 | 29 GB / 約 18 時間 → **約 1.6 GB per audio-hour**（256 次元 content + mel + cache の 768 次元） |
| base 学習 | **8.14 step/s**、130 examples/s、151,354 frames/s（step 1,000 時点） |
| peak VRAM | **1.95 GB**（6 節。24 GB という見積もりは大きく外していた） |
| 30,000 step の所要 | 約 **61 分** |

**外挿の目安（見積もり）:** この条件なら 100 時間の corpus でも抽出は約 5 時間、
学習は step 数で決まります。**ただし `max_batch_size: 16` が効いているので、
batch を増やすと step/s も VRAM も変わります。**

計画値として固定せず、最初の 100 / 1,000 update で examples/sec、frames/sec、peak VRAM、checkpoint size、validation time を計測して再見積もりします。`train.py` がこれを `log/<run>/perf.json` と TensorBoard へ自動で残します。

## 9. 保存すべきデータ台帳

- dataset 名、version、入手元、権利、許可用途、配布可否
- singer ID、曲 ID、take ID、収録条件
- original / separated / denoised の lineage
- checksum と preprocessing manifest
- reject reason と除外前後の時間
- split 作成 seed と split list
- pitch / loudness / duration の分布
