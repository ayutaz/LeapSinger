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
| target similarity | speaker embedding cosine / SECS | singing domain 対応 encoder を比較する |
| intelligibility | CER / phoneme error | 日本語歌唱 ASR の bias を明記する |
| signal quality | SIG / BAK / OVRL、DNSMOS 系 | music/singing への妥当性を過信しない |
| spectral | mel/STFT distance、MCD | parallel reference がある subset に限定 |
| timing | onset/offset deviation | source preservation の確認 |
| compute | RTF、peak VRAM、model size | feature extraction と vocoder を含む/除くを併記 |
| streaming | algorithmic latency、end-to-end latency、boundary error | 実機 audio I/O で測る |

一つの総合点へ早期に集約せず、pitch・内容・target similarity・artifact・latency を別軸で報告します。

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
| streaming | chunk boundary、state reset、buffer under/overrun |
| data | accompaniment leakage、reverb imprint、duplicate leakage |

failure clip は削除せず、category と suspected component を付けます。

## 8. 合格条件

**要ユーザー判断:** 数値閾値は target use case と test set 作成後に確定します。少なくとも以下を満たすまで品質達成を宣言しません。

- 実音声の end-to-end pipeline が再現可能。
- unseen-source / held-out-song で重大な内容崩壊がない。
- Seed-VC baseline に対する blind preference を記録済み。
- target similarity の改善が pitch / intelligibility の悪化だけで得られていない。
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
