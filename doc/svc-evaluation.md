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
| target similarity | speaker embedding cosine / SECS | **手元の x-vector encoder 2 本は歌声で較正を通らない**（下記）。encoder を先に検証すること |
| intelligibility | CER / phoneme error | 日本語歌唱 ASR の bias を明記する |
| signal quality | SIG / BAK / OVRL、DNSMOS 系 | music/singing への妥当性を過信しない |
| spectral | mel/STFT distance、MCD | parallel reference がある subset に限定 |
| **音の明るさ** | **spectral centroid、帯域エネルギー比**（[`tools/audio_metrics.py`](../tools/audio_metrics.py)） | **source ではなく「GT mel をボコーダーに通した再合成」を上限の基準にする。** 内容・音高・V/UV の指標は高域の欠落を検知しない（M3 で実測） |
| timing | onset/offset deviation | source preservation の確認 |
| compute | RTF、peak VRAM、model size | feature extraction と vocoder を含む/除くを併記 |
| streaming | algorithmic latency、end-to-end latency、boundary error | 実機 audio I/O で測る |

一つの総合点へ早期に集約せず、pitch・内容・target similarity・artifact・latency を別軸で報告します。

**確認済み（2026-08-31、M4）: 話者類似度は「まだ測れていない」が正確です。** 枠組みは
[`tools/speaker_similarity.py`](../tools/speaker_similarity.py) にあり、**上限**（target の別クリップ
どうし）・**下限**（無関係な話者）・回復率という、内容保持と同じ読み方をします。しかし
`transformers` の x-vector モデル 2 本を素材を変えて 3 通り較正したところ、最も公平な条件
（同一技法・別の曲の抜粋）でも**同性間で分離できませんでした**。

| 素材 / モデル | 同一話者 | 別話者・同性 | 別話者・異性 | 重なり |
|---|---|---|---|---|
| arpeggios 4 本 / wavlm-base-plus-sv | 0.7533 | 0.6961 | 0.6813 | 88.3% |
| 同上 / unispeech-sat-base-plus-sv | 0.7426 | 0.6970 | 0.6400 | 93.9% |
| **excerpts 3 曲 / wavlm-base-plus-sv** | **0.8307** | **0.7713** | **0.5199** | **83.3%** |

**決定: 同性間の target similarity は、歌声で較正を通る encoder に差し替えるまで報告しません。**
異性間（0.5199 対 0.8307）は分かれるので、粗い確認にのみ使えます。**較正を通していない
encoder の cosine を「話者類似度」として出さないこと。** この表が無ければ、0.77 という数値は
一見それらしく見えます。

**確認済み（2026-08-31、M4）: 明るさを符号つき平均で評価しないこと。** 上限より明るい clip と
暗い clip が打ち消し合い、**平均は良く見えるのに実際は両方向へ外れている**ことが起きます
（実測で範囲 −70% 〜 +33%）。**上限からの距離（絶対値）**で見ます。この誤りで「多 step にすると
明るさが戻る」という結論を一度出しました。また **spectral centroid は測定時の移調条件に強く
依存する**ので、`--transpose` の値を必ず併記します（同じ clip が 0 半音で 540 Hz、
+12 半音で 1196 Hz）。

**確認済み（2026-08-30、M3）: 内容指標は音の劣化を検知しません。** 推論条件の不具合で spectral centroid が 620 → 368 Hz へ落ちたとき、content cos は 0.8217 → 0.8096 としか動かず、F0 相関も V/UV もほぼ無反応でした。**耳で「こもっている」と分かる差です。** 内容・音高の指標が揃って良いことを音質の根拠にしないでください。

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
