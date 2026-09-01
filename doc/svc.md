# LeapSVC 調査・設計ドキュメント索引

最終更新: 2026-09-01

対象ブランチ: `feature/svc`

## 結論

このブランチでは、既存の LeapSinger の「音素・duration・F0 から mel を作る歌声合成（SVS）」を残したまま、事前計算済みの content feature・F0・V/UV・音量から target singer の mel を生成する、オフライン歌声変換（SVC）経路を追加しています。

現時点でモデル、データ契約、学習・評価・単体推論の配線に加え、**実音声から特徴量を作る前処理**（WAV からコマンド 1 本、再実行で bit 一致）、**23 話者・約 18 時間の multi-singer base 事前学習（60,000 step）**、そこからの **target singer（波音リツ）への fine-tune（20,000 step）**まで到達しています。未知の source singer を入れても内容が崩壊しないことを実測しました。実行環境も Python 3.13 と `uv.lock` で固定しています（[実装状況](svc-implementation-status.md) の「実行環境」表）。

話者類似度も測れるようになりました（2026-09-01）。ECAPA-TDNN を 12 秒以上のクリップで使うと事前登録した較正条件を満たし、**fine-tune で target らしさが上がっている**ことを確認しています（回復率 45.1% → 54.9%）。

M5 の客観指標も測定しました（26 clip を両システムで変換）。**話者類似度では Seed-VC が上回り**、事前登録した規則により **「Seed-VC より良い」とは書けません**。一方 **F0 追従と V/UV は LeapSVC が上**で、これは F0 を条件として直接与える設計の表れです。詳細と限界は [実行計画](svc-plan.md) M5 の進捗節。

一方、**音質の評価、Seed-VC との比較、リアルタイム student は未完了**です。したがって「リアルタイム動作」や「Seed-VC より高品質」はまだ確認済みの成果ではありません。**推論 RTF の実測（CPU で合計 0.654）も「リアルタイム」の根拠にはなりません** — chunk 境界・audio I/O・連続運転を測っていないためです。

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
| 話者類似度 | **ECAPA-TDNN を 12 秒以上のクリップで使う。** 較正表（同一話者 / 別話者・同性 / 別話者・異性の重なり）を示さずに cosine を「話者類似度」として報告しない |
| M5 の判定 | **guard rail 方式。** 主要判定は blind preference、客観 4 指標は悪化の検出に使う。**preference で勝っても guard rail を落としていたら「より良い」と書かない** |
| 客観指標の読み方 | **絶対値を品質として報告しない。** 上限（GT mel をボコーダーに通した再合成）との差で読む。CER と信号品質は**上限が無ければ実行そのものを拒否**する |
| checkpoint 選択 | **train loss だけで選ばない。** 選択規則を実験前に config へ書き、未知 source の内容保持が base から一定以上落ちた checkpoint は候補から外す（M4 で実際に発動） |
| 男女をまたぐ変換 | **`--transpose` で source F0 を移調してから変換する**（+7〜+12 半音）。出力のスペクトル傾斜が入力 F0 に強く従うため、移調なしの数値で品質を判断しない |
| 完了判定 | コードの存在ではなく、実音声・比較試聴・遅延実測まで段階別に判定する |

## 未決事項

- speaker embedding を fine-tune 後も残すか、target 固定モデルへ焼き込むか。
- offline teacher の合格閾値と、許容する streaming lookahead / 往復遅延。
- 256 次元部分集合の seed 0 と seed 1 の差（**1 度比較して seed 1 が良い方向だったが、各 1 run では部分集合の差と run のばらつきを分離できない**。**反復は未実施のまま M4 まで進みました**。seed 0 を既定として据え置いています）。
- ~~話者類似度をどう測るか~~ — **決着（2026-09-01）。** ECAPA-TDNN（`speechbrain`、`eval` extra）を **12 秒以上**のクリップで使うと、事前登録した合格条件（重なり 20% 以下）を満たします。較正は [`tools/speaker_calibrate.py`](../tools/speaker_calibrate.py) で再実行できます。**encoder と長さの両方が要ります**（x-vector は 12 秒でも 77.0% で不合格、ECAPA も 6 秒では 56.9% で不合格）。
- ~~話者性を上げる手~~ — **`num_steps` は実施しました（2026-09-01）。** 既定を 16 にして
未知 source の回復率が 60.7% → **69.4%**、timing も 66.2% → **71.9%**（Seed-VC を上回る）。
**ただし話者類似度はまだ Seed-VC に届かず**（0.4910 対 0.5912）、guard rail は落ちたままです。
- ~~GAN を有効にするか~~ — **fine-tune で実施しました（2026-09-02、案 A）。** 未知 source の
回復率が 69.4% → **75.3%**、timing も 73.7% で Seed-VC（70.8%）を上回ります。**過平滑が主因
だったことの確認**になりました。**ただし話者類似度はまだ Seed-VC に届かず**（0.4981 対 0.5912）、
guard rail は落ちたままです。
- **案 B（base から GAN 付きで学習し直す）へ進むか**（**要ユーザー判断**）。案 A で効果は
確認できましたが、**8〜12 時間の学習**になります。
- **入力 F0 に対する出力スペクトル傾斜の結合をモデル側で緩めるか。** 変換時の移調（`--transpose`）で実用上は回避できており、**fine-tune では緩みませんでした**（M4 で実測）。緩めるには**特徴抽出前**の pitch augmentation が要ります（**未実装**。SVC では online `pitch_aug` を使えません）。
- ~~M5 の合格条件と主観評価の条件~~ — **決着（2026-09-01、測定の前に決定）。** 判定は **guard rail 方式**（主要判定は blind preference、客観 4 指標は「Seed-VC より clip 間ばらつきを超えて悪化していないこと」）。主観評価は **N=1 の非公式 preference test**（MOS とは呼ばない）。test set は **20 clip / 12 秒以上 / 日本語を含む / 同性・異性の両方**。詳細は [実行計画](svc-plan.md) M5「事前登録した判定規則」。

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
| **M3 multi-singer base** | **完了**。23 話者 / 約 18 時間を **60,000 step** 学習。**未知 source（VocalSet）の内容保持が学習済み歌手と同等**（下限からの回復率 85.7% 対 84.4%）。継続学習で eval/loss 0.02311 → **0.01459**、1 step と 16 step の乖離 10.8 → **6.0 点**。peak VRAM 2.0 GB |
| **M4 target fine-tune** | **完了**。base から波音リツへ 20,000 step。**target らしさは上がり（話者類似度の回復率 45.1% → 54.9%、自己再構成は上限比 94.8% → 96.8%）、未知 source の内容保持は落ちます**（0.8599 → 0.8440）。事前登録した規則で **`ckpt_010000` を選択**（train loss だけなら 20,000 step を選んでいた）。peak VRAM 2.08 GB |
| **M5 offline 品質ゲート** | **客観指標は測定完了、blind listening test だけ未実施**。**話者類似度で Seed-VC が上回り**（GAN 版で 0.4981 対 0.5912）、guard rail が落ちたため**「Seed-VC より良い」とは書けません**。**F0 追従・V/UV・timing は LeapSVC が上**。GAN fine-tune で未知 source の回復率は 69.4% → **75.3%**（Seed-VC 77.1%） |
| M6 streaming student | 未着手 |

## 完了レベル

1. **実装レベル** — モデル、loader、学習・評価・推論配線が存在する。**現在到達**。
2. **合成 smoke レベル** — 人工テンソルで shape、padding、checkpoint、forward/inference を確認する。**現在到達**。
3. **実データレベル** — 再現可能な前処理で実音声 shard を作り、学習して WAV を生成する。**現在到達**（2026-08-30、M2）。
4. **品質比較レベル** — held-out song と未知 source singer で Seed-VC を含む blind comparison を**完了する**。**勝つことは条件ではありません**（品質を外部基準に対して言えるようになった状態を指します）。**未到達**（2026-09-01 時点で offline teacher・測定手段・判定規則は揃い、**比較そのものが未実施**）。
5. **リアルタイムレベル** — 実機で chunk 境界、RTF、lookahead、総遅延を測り、連続運転する。**未到達**。
