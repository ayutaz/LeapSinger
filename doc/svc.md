# LeapSVC 調査・設計ドキュメント索引

最終更新: 2026-08-30

対象ブランチ: `feature/svc`

## 結論

このブランチでは、既存の LeapSinger の「音素・duration・F0 から mel を作る歌声合成（SVS）」を残したまま、事前計算済みの content feature・F0・V/UV・音量から target singer の mel を生成する、オフライン歌声変換（SVC）経路を追加しています。

現時点でモデル、データ契約、学習・評価・単体推論の配線と合成入力によるテストまでは実装済みです。一方、実音声から特徴量を作る前処理、実データ学習、Seed-VC との音質比較、リアルタイム student は未完了です。したがって「リアルタイム動作」や「Seed-VC より高品質」はまだ確認済みの成果ではありません。

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
| [実行計画（マイルストーン）](svc-plan.md) | M0〜M6 の目的・ゴール・完了条件、条件付きトラック、進行ルール |
| [出典一覧](svc-sources.md) | 一次資料、ローカル根拠、調査時点 |

## 主要な決定事項

| 項目 | 決定 |
|---|---|
| 製品目標 | zero-shot 万能モデルではなく、追加学習可能な target-singer SVC |
| 最初の品質目標 | まず offline teacher を完成させ、Seed-VC と比較する |
| streaming | offline 品質ゲート通過後に causal / limited-lookahead student を蒸留する |
| 既存 SVS | 削除せず、`model.arch: svc` の別経路として維持する |
| content encoder | ContentVec / HuBERT 系の事前学習済み表現を凍結して使う方針 |
| pitch | RMVPE 等で source F0 と V/UV を抽出する方針 |
| vocoder | NHVSing と互換な 44.1 kHz・hop 256・128-bin ln-mel を出力する |
| 学習順序 | multi-singer base の事前学習を推奨し、target singer へ fine-tune |
| 完了判定 | コードの存在ではなく、実音声・比較試聴・遅延実測まで段階別に判定する |

## 未決事項

- ContentVec と HuBERT のどの層・モデルを正式採用するか。
- F0 extractor を RMVPE に固定するか、SwiftF0 等も比較するか。
- target singer と multi-singer corpus の権利条件を満たせるか。
- speaker embedding を fine-tune 後も残すか、target 固定モデルへ焼き込むか。
- offline teacher の合格閾値と、許容する streaming lookahead / 往復遅延。
- NHVSing の既存重みをそのまま使うか、target singer へ追加学習するか。

## 完了レベル

1. **実装レベル** — モデル、loader、学習・評価・推論配線が存在する。**現在到達**。
2. **合成 smoke レベル** — 人工テンソルで shape、padding、checkpoint、forward/inference を確認する。**現在到達**。
3. **実データレベル** — 再現可能な前処理で実音声 shard を作り、学習して WAV を生成する。**未到達**。
4. **品質比較レベル** — held-out song と未知 source singer で Seed-VC を含む blind comparison を完了する。**未到達**。
5. **リアルタイムレベル** — 実機で chunk 境界、RTF、lookahead、総遅延を測り、連続運転する。**未到達**。
