# LeapSVC 調査・設計ドキュメント索引

最終更新: 2026-08-30

対象ブランチ: `feature/svc`

## 結論

このブランチでは、既存の LeapSinger の「音素・duration・F0 から mel を作る歌声合成（SVS）」を残したまま、事前計算済みの content feature・F0・V/UV・音量から target singer の mel を生成する、オフライン歌声変換（SVC）経路を追加しています。

現時点でモデル、データ契約、学習・評価・単体推論の配線に加え、**実音声から特徴量を作る前処理**（WAV からコマンド 1 本、再実行で bit 一致）と、**実音声を overfit して WAV を出すところ**まで到達しています。実行環境も Python 3.13 と `uv.lock` で固定し、CUDA 版 torch が実機で動くところまで確認しました（[実装状況](svc-implementation-status.md) の「実行環境」表）。一方、汎化する実データ学習、Seed-VC との音質比較、リアルタイム student は未完了です。したがって「リアルタイム動作」や「Seed-VC より高品質」はまだ確認済みの成果ではありません。

## 記述ラベル

各文書では、内容の確度を次のように区別します。

| ラベル | 意味 |
|---|---|
| **確認済み** | ローカルのコード・設定・テスト、または一次資料で確認した事実 |
| **決定** | この開発で採用する方針 |
| **推奨** | 現時点の技術判断。実験結果で変更し得るもの |
| **見積もり** | 実測前のデータ量・GPU・時間の目安 |
| **仮説** | 評価実験で検証すべき品質上の予想 |
| **未実装** | 設計はあるがコードまたはデータがないもの |
| **要ユーザー判断** | データ権利、目標品質、遅延など、利用者の決定が必要なもの |

## 文書一覧

| 文書 | 内容 |
|---|---|
| [プロジェクトと周辺エコシステム](project-ecosystem.md) | LeapSinger と作者の関連リポジトリ、現行 SVS の能力と限界 |
| [SVC 要件](svc-requirements.md) | 目的、非目標、入出力、品質ゲート、決定事項 |
| [SVC アーキテクチャ](svc-architecture.md) | 特徴量、harmonic prior、rectified flow、offline/streaming 構成 |
| [データセットと計算資源](svc-data-compute.md) | 音声条件、必要量の目安、GPU/時間の見積もり |
| [学習計画](svc-training.md) | 多歌手事前学習、target fine-tune、warm-start、distillation |
| [評価計画](svc-evaluation.md) | 比較対象、客観・主観評価、ablation、合格条件 |
| [先行研究・ライセンス・リスク](svc-prior-art-license.md) | 新規性の境界、ライセンス、技術・データ上のリスク |
| [実装状況と再現手順](svc-implementation-status.md) | 現在のファイル、データ契約、実行例、検証済み/未検証境界 |
| [データセット台帳](svc-dataset-ledger.md) | 既存 3 DB の権利条件（規約 URL と取得日つき）、SVC で新たに要るもの |
| [実行計画（マイルストーン）](svc-plan.md) | M0〜M6 の目的・ゴール・完了条件、条件付きトラック、進行ルール |
| [content encoder の選定](svc-content-encoder.md) | 候補一覧とライセンス、Interspeech 2025 の比較実証、先行実装の採用状況 |
| [出典一覧](svc-sources.md) | 一次資料、ローカル根拠、調査時点 |

## 主要な決定事項

| 項目 | 決定 |
|---|---|
| 製品目標 | zero-shot 万能モデルではなく、追加学習可能な target-singer SVC |
| 最初の品質目標 | まず offline teacher を完成させ、Seed-VC と比較する |
| streaming | offline 品質ゲート通過後に causal / limited-lookahead student を蒸留する |
| 既存 SVS | 削除せず、`model.arch: svc` の別経路として維持する |
| content encoder | **ContentVec**（`lengyue233/content-vec-best`、MIT、768 次元 layer 12）を凍結して使う。学習は 768 から固定ランダムに選んだ **256 次元**を既定とする（[根拠](svc-content-encoder.md)） |
| pitch | **RMVPE に固定**して source F0 と V/UV を抽出する |
| vocoder | NHVSing と互換な 44.1 kHz・hop 256・128-bin ln-mel を出力する |
| 学習順序 | multi-singer base の事前学習を推奨し、target singer へ fine-tune |
| 学習環境 | 学習は vast.ai の Linux GPU インスタンスで行う。手元の Windows 機は開発・推論・検証用 |
| 用途と配布 | **研究・個人利用のみ、モデルの配布なし**。非商用ライセンスのコーパスも使える（[台帳](svc-dataset-ledger.md)） |
| 素材 | base は **GTSinger**（80.6 h / 20 歌手 / 48 kHz / 実録音）、未知 source の test は **VocalSet**、target は既存 3 DB |
| 完了判定 | コードの存在ではなく、実音声・比較試聴・遅延実測まで段階別に判定する |

## 未決事項

- speaker embedding を fine-tune 後も残すか、target 固定モデルへ焼き込むか。
- offline teacher の合格閾値と、許容する streaming lookahead / 往復遅延。
- 256 次元部分集合の seed 0 と seed 1 の差（M3 の前に 1 度だけ確かめる）。

**決着済み（2026-08-30）:** 補間方法は **left（直前保持）**、256 次元の部分集合は **seed 0 を既定**、
`content-vec-best` は **layer 12 の hidden state（768 次元・50 Hz）を実モデルで確認**、
**NHVSing は既存重みをそのまま使う**（GTSinger に out-of-distribution なペナルティは
観測されなかった。[台帳 7b 節](svc-dataset-ledger.md#7b-nhvsing-にとって-in-distribution-かの実測2026-08-30)）、
用途は**研究・個人利用のみ**、素材は **target = 波音リツ / base = GTSinger + 日本語 3 DB /
未知 source test = VocalSet** に確定しました（[content encoder の選定](svc-content-encoder.md)、
[データセット台帳](svc-dataset-ledger.md)）。

## マイルストーンの進捗

[実行計画](svc-plan.md) の M0〜M6 に対する現在地です。

| | 状態 |
|---|---|
| **M0 データ確定** | **入手可能な素材について完了**。5 コーパスを取得して検査・coverage・split を実データで実行。台帳に権利・lineage・checksum・確定した割り当てを記録。東北きりたん / No.7 はログイン必須で未取得 |
| **M1 特徴抽出前処理** | **完了**。WAV から `svc_shard.npz` を 1 コマンドで作り、再実行で bit 一致。実音声で全ゴール検証済み |
| **M2 実音声 smoke** | **完了**。vast.ai で overfit し WAV を生成。F0 相関 0.9991、決定的モードで bit 再現。**完了レベル 3 に到達** |
| M3〜M6 | 未着手。M3 の前に **256 次元 seed 0 / 1 の比較**（M1 決定 2）だけ残っています |

## 完了レベル

1. **実装レベル** — モデル、loader、学習・評価・推論配線が存在する。**現在到達**。
2. **合成 smoke レベル** — 人工テンソルで shape、padding、checkpoint、forward/inference を確認する。**現在到達**。
3. **実データレベル** — 再現可能な前処理で実音声 shard を作り、学習して WAV を生成する。**現在到達**（2026-08-30、M2）。
4. **品質比較レベル** — held-out song と未知 source singer で Seed-VC を含む blind comparison を完了する。**未到達**。
5. **リアルタイムレベル** — 実機で chunk 境界、RTF、lookahead、総遅延を測り、連続運転する。**未到達**。
