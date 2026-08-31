# LeapSVC 評価計画

## 1. 評価原則

- train reconstruction と conversion quality を分ける。
- target singer と同じ歌手の自己再構成だけで合格にしない。
- held-out song と未知 source singer を必須にする。
- 同一 source clip、同一 target、同一 loudness / silence 処理で比較する。
- 数値指標だけでなく blind listening test と failure analysis を残す。
- 選んだ成功例だけでなく、全評価 clip の manifest と出力を保存する。

## 2. 比較対象

| ID | システム | 目的 |
|---|---|---|
| A | source vocal | 内容・F0・タイミングの基準 |
| B | target ground truth | target timbre の上限参考。平行録音がなければ別 phrase |
| C | Seed-VC SVC baseline | 公開 SVC との比較 |
| D | LeapSVC target-only | multi-singer pretraining の効果を測る |
| E | LeapSVC base -> target fine-tune | 主提案 |
| F | E + target-fine-tuned NHVSing | vocoder fine-tune の寄与を測る |
| G | streaming student | offline teacher からの品質低下と遅延改善を測る |

Seed-VC は archived repository の最新固定 revision、checkpoint、inference steps、F0 条件、reference clip を manifest に保存します。default と高速設定を混同しません。

## 3. test set

少なくとも次の軸を交差させます。

- source singer: base 学習済み / 未知
- source pitch: target range 内 / 上限付近 / 外側
- phonetic content: slow / fast / 子音密度が高い
- phonation: chest / falsetto / breathy / strong / weak
- expression: long tone / vibrato / slide
- recording: clean isolated / separated stem（使用する場合）

target singer の train 曲、同一 take、近重複 clip は test から除外します。

## 4. 客観指標

| 観点 | 候補指標 | 注意 |
|---|---|---|
| pitch 保持 | F0 correlation、F0 RMSE、V/UV error | extractor 自身の誤りを別途監査する |
| target similarity | speaker embedding cosine / SECS | **ECAPA-TDNN を 12 秒以上のクリップで使う**（下記の較正表）。encoder と長さの**両方**が要る |
| intelligibility | CER / phoneme error | 日本語歌唱 ASR の bias を明記する |
| signal quality | SIG / BAK / OVRL、DNSMOS 系 | music/singing への妥当性を過信しない |
| spectral | mel/STFT distance、MCD | parallel reference がある subset に限定 |
| **音の明るさ** | **spectral centroid、帯域エネルギー比**（[`tools/audio_metrics.py`](../tools/audio_metrics.py)） | **source ではなく「GT mel をボコーダーに通した再合成」を上限の基準にする。** 内容・音高・V/UV の指標は高域の欠落を検知しない（M3 で実測） |
| timing | onset/offset deviation | source preservation の確認 |
| compute | RTF、peak VRAM、model size | feature extraction と vocoder を含む/除くを併記 |
| streaming | algorithmic latency、end-to-end latency、boundary error | 実機 audio I/O で測る |

一つの総合点へ早期に集約せず、pitch・内容・target similarity・artifact・latency を別軸で報告します。

**確認済み（2026-09-01）: 話者類似度は測れるようになりました。encoder とクリップ長の両方が要ります。**
枠組みは [`tools/speaker_similarity.py`](../tools/speaker_similarity.py)、較正は
[`tools/speaker_calibrate.py`](../tools/speaker_calibrate.py) で**再実行できます**（VocalSet
20 歌手・excerpts/straight・各 3 本）。**上限**（target の別クリップどうし）・**下限**
（無関係な話者）・回復率という、内容保持と同じ読み方をします。

**合格条件は走らせる前に決めました:** 重なり（同一話者ペアの下位 5% を超える別話者・同性
ペアの割合）が **20% 以下**。

| encoder | クリップ長 | 同一話者 | 別話者・同性 | 別話者・異性 | 重なり | 判定 |
|---|---:|---|---|---|---:|:--:|
| wavlm-base-plus-sv | 6 s | 0.8307 ± 0.0832 | 0.7713 ± 0.1104 | 0.5199 ± 0.1296 | 83.3% | 不合格 |
| wavlm-base-plus-sv | 12 s | 0.8856 ± 0.0670 | 0.8196 ± 0.0943 | 0.5314 ± 0.1323 | 77.0% | 不合格 |
| ECAPA-TDNN | 6 s | 0.5415 ± 0.1331 | 0.3452 ± 0.1325 | 0.1816 ± 0.0976 | 56.9% | 不合格 |
| **ECAPA-TDNN** | **12 s** | 0.6632 ± 0.0995 | 0.3891 ± 0.1369 | 0.1966 ± 0.1013 | **19.8%** | **合格** |
| **ECAPA-TDNN** | **20 s** | 0.7096 ± 0.1037 | 0.4040 ± 0.1386 | 0.2109 ± 0.1096 | **17.3%** | **合格** |

**どちらか一方では足りません。** x-vector は 12 秒にしても 77.0% で不合格、ECAPA も 6 秒では
56.9% で不合格です。**2026-08-31 に「測れない」と結論したのは、6 秒で切って測っていたことが
原因の半分でした。** クリップ長は encoder の選択と同じ重みを持つ設計パラメータとして扱います。

**決定: 較正表を示さずに cosine を「話者類似度」として出さないこと。** また
`similarity_report` は 12 秒未満のクリップを**拒否**します（較正の外だから）。

**確認済み（2026-08-31、M4）: 明るさを符号つき平均で評価しないこと。** 上限より明るい clip と
暗い clip が打ち消し合い、**平均は良く見えるのに実際は両方向へ外れている**ことが起きます
（実測で範囲 −70% 〜 +33%）。**上限からの距離（絶対値）**で見ます。この誤りで「多 step にすると
明るさが戻る」という結論を一度出しました。また **spectral centroid は測定時の移調条件に強く
依存する**ので、`--transpose` の値を必ず併記します（同じ clip が 0 半音で 540 Hz、
+12 半音で 1196 Hz）。

**確認済み（2026-08-30、M3）: 内容指標は音の劣化を検知しません。** 推論条件の不具合で spectral centroid が 620 → 368 Hz へ落ちたとき、content cos は 0.8217 → 0.8096 としか動かず、F0 相関も V/UV もほぼ無反応でした。**耳で「こもっている」と分かる差です。** 内容・音高の指標が揃って良いことを音質の根拠にしないでください。

### M5 に入る前の棚卸し（2026-09-01）

**確認済み:** [実行計画](svc-plan.md) M5 ゴール 2 が挙げる指標のうち、**道具があるのは半分**です。
M5 は「測る」工程なので、無い指標は M5 の中で作ることになります。**先に把握しておくための表**です。

| M5 が要求する指標 | 道具 | 状態 |
|---|---|---|
| F0 correlation / RMSE | [`m3_verify.py`](../tools/m3_verify.py)（`f0_corr` / `median_semitones`） | **ある** |
| V/UV error | 同上（`uv_agree`） | **ある** |
| 内容保持（明瞭度の代理） | 同上（`content_cos` を上限・下限つきで） | **ある** |
| 音の明るさ・帯域 | [`audio_metrics.py`](../tools/audio_metrics.py) + 上限（GT mel 再合成） | **ある** |
| speaker similarity | [`speaker_similarity.py`](../tools/speaker_similarity.py) + [較正](../tools/speaker_calibrate.py) | **ある**（2026-09-01 に較正通過） |
| **CER / 音素誤り** | — | **無い。** 日本語歌唱 ASR が要る（依存の追加。bias の明記も要る） |
| **信号品質（DNSMOS 系）** | — | **無い。** 歌声への妥当性そのものが要検討 |
| **timing（onset/offset ずれ）** | — | **無い** |
| **RTF / peak VRAM（推論時）** | — | **無い。** 学習側の peak VRAM は測ってあるが、推論の RTF は未測定 |

**注意:** `content_cos` は明瞭度の代理であって CER ではありません。M3 で、耳で分かる劣化
（centroid 620 → 368 Hz）に対して content cos は 0.8217 → 0.8096 としか動きませんでした。
**内容指標が揃って良いことを「明瞭度が保たれた」の根拠にしないこと。**

## 5. 主観評価

### Blind test の質問

1. どちらが target singer に似ているか。
2. どちらが歌詞を聞き取りやすいか。
3. どちらが自然で、buzz、metallic、phase、breath artifact が少ないか。
4. source の音程と表現をどちらが保っているか。
5. 総合的にどちらを選ぶか。

### 実施条件

- system 名を隠し、順序を randomize する。
- loudness を揃え、無音長や file name から system が分からないようにする。
- headphone / speaker、評価者の歌唱・音響経験を記録する。
- 1 clip あたりの全比較数を制限し、疲労を管理する。
- tie / 判定不能を許可する。
- 評価者数、clip 数、信頼区間、除外規則を事前に決める。

## 6. ablation

- ContentVec vs HuBERT、encoder layer、feature normalization
- UV あり/なし
- loudness あり/なし、正規化方式
- speaker condition あり/なし
- harmonic/noise prior vs random/noise prior
- 1-step vs 複数 step
- SVS warm-start vs independent initialization
- target-only vs multi-singer base
- pitch/formant augmentation あり/なし
- GAN あり/なし
- NHVSing frozen vs target fine-tune
- full-context teacher vs causal / limited-lookahead student

一度に複数要素を変えず、同一 split・seed・更新 budget を使います。

## 7. failure taxonomy

| 分類 | 例 |
|---|---|
| content | 子音欠落、母音化、歌詞置換 |
| pitch | octave error、裏声で drop、vibrato 平滑化 |
| timbre | source leakage、target identity 不足、性別/formant 不整合 |
| dynamics | 強弱消失、breath 過多、loudness pumping |
| vocoder | buzz、metallic、high-frequency noise、クリック |
| timing | onset 遅延、子音の先頭欠落、phrase 末尾切れ |
| **音域外** | **source が C3（約 131 Hz）より低いときの崩れ。** 手元の素材はこの帯域が最大でも 3.6% しかなく（[データセット台帳](svc-dataset-ledger.md) 4b 節）、低い男声 source は学習分布の外側への外挿になる。**必ず観察項目に入れる** |
| streaming | chunk boundary、state reset、buffer under/overrun |
| data | accompaniment leakage、reverb imprint、duplicate leakage |

failure clip は削除せず、category と suspected component を付けます。

## 8. 合格条件

**要ユーザー判断:** 数値閾値は target use case と test set 作成後に確定します。少なくとも以下を満たすまで品質達成を宣言しません。

- 実音声の end-to-end pipeline が再現可能。
- unseen-source / held-out-song で重大な内容崩壊がない。
- Seed-VC baseline に対する blind preference を記録済み。
- target similarity の改善が pitch / intelligibility の悪化だけで得られていない。
  **前提として、使う speaker encoder が上限・下限・重なりの較正を歌声で通っていること**（4 節）。
- 未知 source の内容保持が base から落ちていないこと。**M4 で、target の再現と未知 source の
  保持が単調に逆へ動くことを実測しました。**片方だけの改善を品質向上と呼ばないこと。
- streaming では teacher との差と実測遅延を同時に提示できる。

## 9. 報告テンプレート

各評価 report に次を含めます。

```text
date / commit / dirty state
systems and exact checkpoints
dataset and test manifest
preprocessing revisions
hardware / software / seed
objective metrics with per-group breakdown
blind-test protocol and results
failure counts and representative samples
confirmed conclusions
inconclusive observations
next experiment
```
