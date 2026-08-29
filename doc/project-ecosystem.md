# LeapSinger と周辺エコシステム

調査時点: 2026-08-30

## 作者の公開リポジトリ

**確認済み:** wavtechyukky の GitHub 公開リポジトリ一覧には、現在次の 5 件があります。

| リポジトリ | 役割 | SVC 開発との関係 |
|---|---|---|
| [LeapSinger](https://github.com/wavtechyukky/LeapSinger) | 音素・duration・F0 から mel を作る 1-step rectified-flow 歌声合成モデル | 本開発の土台 |
| [NHVSing](https://github.com/wavtechyukky/NHVSing) | mel と F0 から波形を生成する Neural Homomorphic Vocoder | SVC 出力を WAV にする候補 |
| [pyshiro](https://github.com/wavtechyukky/pyshiro) | HMM/HSMM ベースの音素アライメント | 既存 SVS データ作成には有用。WAV-only SVC では必須ではない |
| [realigned-singing-labels](https://github.com/wavtechyukky/realigned-singing-labels) | 歌声データセット用の再アライメント済みラベル | SVS 用ラベル資産。音声自体は含まない |
| [pitch-benchmark](https://github.com/wavtechyukky/pitch-benchmark) | 複数 pitch extractor の比較 | SVC の F0 extractor 選定資料 |

```text
既存 SVS:
音素 + duration + F0 -> LeapSinger -> mel + F0 -> NHVSing -> WAV

追加する SVC:
source WAV -> content / F0 / UV / loudness -> LeapSVC -> mel + F0 -> NHVSing -> WAV
```

## LeapSinger の現在の能力

**確認済み:** 既存 LeapSinger は text-only モデルでも WAV-to-WAV 変換器でもありません。フレームへ展開できる音素情報、duration、F0 を受け、mel を生成します。

| 機能 | 状態 | 根拠・注意 |
|---|---|---|
| 1-step rectified flow | 実装済み | F0 由来の harmonic/noise 擬似 mel を始点にする |
| V/UV conditioning | 実装済み | 無声部で harmonic 成分を gate できる |
| 多話者 | 実装済み | speaker ID / embedding を条件にできる |
| スタイル | 実装済み | 同一話者内の style 切り替えを想定 |
| NHVSing 連携 | 実装済み | 44.1 kHz、hop 256 の構成を含む |
| ONNX export | 実装済み | 主に既存 SVS 経路向け |
| OpenUTAU export | 実験的 | README 自身が実機未検証と明記 |
| 直接 SVC | 元ブランチにはない | 本ブランチで追加中 |

LeapSinger README に記載された音響モデルの CPU RTF は、Apple Silicon 10 コア・約 7 秒のフレーズで Python native 1 コアが 0.027 です。これは作者の条件での既存 **SVS 音響モデル**の測定値であり、SVC の特徴抽出、ボコーダー、ストリーミング I/O を含む end-to-end 遅延ではありません。

## 周辺ツールから得られる示唆

### pyshiro と realigned-singing-labels

pyshiro は音素境界を作るための HSMM/HMM 系ツールです。既存 SVS の教師データには役立ちますが、今回の SVC は source/target の WAV から自己教師あり content feature、F0、音量を自動抽出するため、歌詞・音素ラベル・MIDI を必須にしない設計です。

### pitch-benchmark

公式 README の集計では SwiftF0 の平均値が 90.2、RMVPE が 87.2、CREPE が 85.3 とされています。一方、人間の歌唱セットでは RMVPE が MIR-1K 96.0、Vocadito 96.4 で最良と報告されています。これは採用候補を絞る参考値であり、本プロジェクトの歌手・録音条件での再評価が必要です。

## なぜ SVC を別経路として追加するか

既存 SVS encoder は「音素を duration に従って展開した条件」を作ります。SVC は「source WAV から抽出した frame-aligned content feature」を条件にします。入力の意味と shape が異なるため、既存の音素 encoder を無理に置換せず、以下を共有するのが安全です。

- harmonic/noise excitation
- rectified-flow backbone
- mel loss / flow loss / optional GAN
- speaker conditioning の仕組み
- checkpoint の基本形式
- NHVSing 互換の mel 出力

**決定:** 既存 SVS は保持し、SVC は `model.arch: svc` で明示的に選ぶ別モデルとして開発します。
