# SVC 先行研究・ライセンス・リスク

調査時点: 2026-08-30

## 1. 先行研究との境界

次の要素は、それぞれ既に公開例があります。

| 要素 | 公開例 | 本開発への意味 |
|---|---|---|
| rectified flow + SVC | [RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC)、[DAFMSVC](https://arxiv.org/abs/2508.05978) | rectified-flow SVC 自体は新規とは言えない |
| harmonic / source-filter modelling + SVC | [Poly-SVC](https://arxiv.org/abs/2605.12310)、[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)、[Source-Filter SVC](https://arxiv.org/abs/2201.10130) | harmonic modelling の利用自体も既知 |
| speaker-disentangled SSL content | [ContentVec](https://arxiv.org/abs/2204.09224) | content feature の source timbre 低減を狙う根拠 |
| robust vocal pitch extraction | [RMVPE](https://arxiv.org/abs/2306.15412) | source F0 条件の候補 |
| harmonic vocoder | [Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.html) | NHVSing 系 vocoder の研究背景 |

本開発で検証する特徴的な組み合わせは、**source F0 から構成した harmonic/noise pseudo mel を rectified flow の `x0` とし、content・F0・UV・loudness 条件で 1-step target-mel transport を行うこと**です。

ただし、同一構成の不存在を網羅的に証明したわけではありません。論文・特許・未索引実装を含む広範な prior-art search なしに「世界初」「唯一」と表現しません。新規性より、速度・品質・target fine-tune 効率を同一条件で実証することを優先します。

## 2. Seed-VC baseline

[Seed-VC](https://github.com/Plachtaa/seed-vc) は zero-shot voice conversion / singing voice conversion を扱う公式実装で、repository は GPL-3.0、2025-11-21 に archive されています。公式 README は SVC 向け 44.1 kHz model、F0-conditioned inference、複数 diffusion step の品質/速度設定を説明しています。

**決定:** Seed-VC のコードを本リポジトリへ取り込むのではなく、外部 baseline として同一入力で比較します。GPL code のコピーや linking を避け、比較 script / manifest だけを独立管理します。

Seed-VC は zero-shot を主眼に含みますが、本開発は target fine-tune を許容します。そのため単なる機能表ではなく、同じ target data budget と実運用条件を明記して比較します。

## 3. ライセンス一覧

| 対象 | 確認内容 | 実務上の扱い |
|---|---|---|
| LeapSinger code | ローカル `LICENSE` は MIT | 本ブランチの派生コードも既存 notice を維持 |
| bundled NHVSing ONNX | LeapSinger README は MIT 対象外と明記 | 配布物ごとの規約を確認 |
| LeapSinger release models / training data | README は MIT 対象外と明記 | `CREDITS.txt` と各 DB 規約を確認 |
| [NHVSing repository](https://github.com/wavtechyukky/NHVSing) | repository code は MIT | code と weight を分けて扱う |
| NHVSing V3 distributed weights | 公式 README は非商用 dataset で学習され商用不可と説明 | 商用製品へそのまま組み込まない |
| [pyshiro](https://github.com/wavtechyukky/pyshiro) | GPL-3.0 | 呼び出し・同梱・派生の境界を法務確認 |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) | GPL-3.0 | baseline として外部実行し、コードを混在させない |

これは法的助言ではありません。実際の配布・商用利用前に、正確な weight file、dataset version、出力音声規約、依存関係を対象として確認します。

## 4. データ権利

歌声データは「閲覧/再生できる」ことと「機械学習、変換モデル配布、生成音声商用利用が可能」なことが別です。dataset ledger へ以下を保存します。

- 原著作者、実演家、録音物、作詞作曲、キャラクター/voicebank 規約
- 学習可否、fine-tune 可否、重み配布可否、生成物利用条件
- commercial / noncommercial、attribution、share-alike、禁止用途
- source separation を行う権利と派生 stem の扱い
- 規約 URL、取得日、version、同意記録

**要ユーザー判断:** target singer と corpus が決まるまで、配布可能性や商用利用可能性は未確定です。

## 5. 技術リスク

### SSL content の timbre leakage

source identity が content に残り、target timbre と混ざる可能性があります。

対策候補: encoder/layer 比較、pitch/formant augmentation、speaker-adversarial loss、content retrieval、unknown-source test。

### F0 error

octave error、breathy/falsetto の unvoiced 判定、高音上限で harmonic prior が誤ります。

対策候補: extractor 比較、confidence の保存、manual audit、range-aware postprocess。ただし恣意的な correction を test だけに適用しません。

### Loudness / recording leakage

マイク gain、compression、room reverb を表現として学習する可能性があります。

対策候補: normalization ablation、recording-session split、augmentation、clean/separated corpus の分離。

### Domain mismatch

speech-pretrained SSL encoder、target singer の声域、NHVSing の学習分布が一致しない可能性があります。

対策候補: singing-domain evaluation、ground-truth-mel vocoder test、target vocoder fine-tune の分離比較。

### Streaming degradation

offline model の future context を除くと子音、vibrato、phrase boundary が悪化し得ます。

対策候補: limited lookahead、teacher distillation、state cache、chunk augmentation、境界専用評価。

## 6. 研究上の主張ルール

- 「確認済み」はコード、実行 artifact、一次資料のいずれかを示す。
- 「Seed-VC より良い」は同一 test set の blind comparison 後にのみ使う。
- 「リアルタイム」は対象 hardware の end-to-end latency と連続動作後にのみ使う。
- 「少量データ」は時間だけでなく歌手数、coverage、権利、split を併記する。
- 「1-step」は acoustic flow の step 数であり、全 pipeline が 1 operation という意味ではない。
