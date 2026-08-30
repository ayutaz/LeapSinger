# LeapSVC アーキテクチャ

## 1. 全体像

```text
source WAV
   |
   +--> frozen SSL encoder --------> content [T, C]
   +--> pitch extractor -----------> F0 [T], V/UV [T]
   +--> loudness extractor --------> loudness [T]
                                          |
                                          v
                                ContentAdapter / condition
                                          |
source F0 + V/UV -> harmonic/noise excitation -> pseudo mel x0
                                          |          |
                                          +----------+
                                                     v
                                         1-step rectified flow
                                                     |
                                                     v
                                           target mel [128, T]
                                                     |
                                             NHVSing + source F0
                                                     |
                                                     v
                                               converted WAV
```

この構成の中心仮説は、source F0 から作った周期構造を持つ擬似 mel を flow の始点 `x0` にすることで、ランダムノイズから周期性を学ぶ負担を減らし、1-step でも target mel へ移送できるというものです。

## 2. 特徴量

### Content

**決定:** ContentVec または HuBERT 系の事前学習済み encoder を凍結し、その出力を前処理で保存します。SVC model の学習 graph 内では SSL encoder を実行しません。

**決定:** ContentVec（`lengyue233/content-vec-best`、MIT、768 次元、layer 12）を採用し、学習には 768 から固定ランダムに選んだ 256 次元を使います。候補比較と根拠は [content encoder の選定](svc-content-encoder.md)。

**未決:** mel grid への resampling 方法（SSL の 50 Hz と mel の 172.265625 Hz は整数比にならない）と、256 次元部分集合の選び方。いずれも再現性に直結するため manifest へ記録します。

**リスク:** SSL feature に source timbre が残ると、target singer への変換を妨げます。pitch/formant augmentation、speaker-adversarial loss、feature retrieval、encoder/layer の比較を候補とします。

### F0 と V/UV

F0 は音程保持と harmonic prior の両方に使います。無声区間は `uv=0` とし、flow 条件と excitation gate に使います。`f0_interp` は harmonic phase を連続にするため gap-filled 値を保持しますが、無声区間を有声として扱う意味ではありません。

**推奨:** 最初は RMVPE を基準にし、target data の高音・裏声・breathy voice で error analysis を行います。pitch-benchmark の一般集計だけで extractor を確定しません。

### Loudness

source の強弱を保持する条件です。raw RMS では録音 gain やマイク差を学習する可能性があるため、log-RMS、窓幅、floor、正規化単位を固定します。

**未決:** phrase 単位正規化か dataset/global 統計か。target dynamics の保持と source recording leakage の trade-off を比較します。

## 3. ContentAdapter と condition

実装済み `ContentAdapter` は `[B, T, content_dim]` の content、log2 F0、V/UV、loudness を共通 hidden width へ写像します。speaker conditioning が有効な場合は speaker embedding も加えます。

既存の phoneme encoder と役割は似ていますが、入力表現が異なるため重みを直接共有しません。adapter の出力を既存 flow backbone が受け取れる shape に合わせることで、flow、excitation、loss の再利用を可能にします。

## 4. Harmonic/noise prior

LeapSinger の既存 excitation を再利用し、source F0 と V/UV から harmonic/noise 擬似 mel を作ります。

```text
F0 -> phase-continuous harmonics -> impulse-like excitation
V/UV ---------------------------> harmonic gate
noise_ratio --------------------> unvoiced/noise component
waveform-like excitation -> STFT/mel -> pseudo mel x0
```

**確認済み:** 現行設定は `n_harm: 256`、`noise_ratio: 0.05`、`exc_scale: 0.15`、`harm_decay: 1.0` です。

**仮説:** pitch 構造を最初から持つ `x0` は 1-step transport を助けます。ただし、harmonic modelling や rectified-flow SVC 自体は先行研究があり、この組み合わせだけから新規性を断定しません。

## 5. Rectified flow と mel

condition と `x0` を使い、1-step で target mel を推定します。初期学習は flow loss と mel reconstruction loss を使い、安定後に optional GAN を検討します。

**決定:** 最初の実音声 smoke と base training では GAN を無効にし、再構成と flow の成立を先に確認します。artifact や不安定化を分類できる状態になってから GAN を追加します。

出力は NHVSing V3 互換を基準とします。

- sample rate: 44,100 Hz
- hop size: 256
- FFT / window: 2,048
- mel bins: 128
- frequency range: 40–16,000 Hz
- representation: repository の既存 ln-mel 定義

## 6. Offline teacher と streaming student

### Offline teacher

現在のモデルは phrase 全体または十分な文脈を持つ non-causal model です。品質上限と failure mode を確認する基準にします。

### Streaming student

**未実装:** causal convolution、limited lookahead、state cache、chunked feature extraction、cross-fade、distillation を備えた student。

候補の蒸留対象:

- teacher mel / flow velocity
- hidden condition
- spectral envelope
- F0-conditioned harmonic structure
- multi-resolution STFT or waveform perceptual outputs

streaming の end-to-end latency は、model RTF だけではなく次の合計です。

```text
audio input buffer
+ SSL encoder receptive field / chunk
+ pitch extractor lookahead
+ acoustic student lookahead
+ vocoder chunk / overlap
+ output buffer
```

## 7. 話者条件

multi-singer base では speaker ID / embedding を使い、話者ごとの mel を自己再構成します。target fine-tune 後は次の二案があります。

1. speaker embedding を残して target ID を固定する。
2. embedding を焼き込み、speaker input のない target-specific model にする。

**推奨:** base 学習中は話者条件を残し、target fine-tune と配布時に固定化を検討します。これにより content / pitch / loudness と speaker identity の分離を学びやすくします。

## 8. 既存 SVS との境界

| 部品 | SVS | SVC | 共有 |
|---|---|---|---|
| 入力 | phoneme、duration、F0 | content、F0、UV、loudness | F0 の概念のみ |
| condition encoder | phoneme encoder | ContentAdapter | 出力 hidden width |
| excitation | F0 harmonic/noise | source F0 harmonic/noise | 共有 |
| flow backbone | rectified flow | rectified flow | 共有 |
| mel / GAN loss | 使用 | 使用可能 | 共有 |
| vocoder | NHVSing | NHVSing | 共有 |

SVC 追加によって既存 SVS の前処理・辞書・export 契約は変更しません。
