# SVC 調査の出典一覧

調査日: 2026-08-30（ローカル根拠は 2026-09-01 まで追記）

## 1. 証拠の扱い

1. ローカルの実装・設定・テスト結果を、現在のブランチ状態の根拠とします。
2. 論文、公式 repository、作者 README を技術仕様・ライセンスの一次資料として優先します。
3. データ量、GPU、学習時間は一次資料で実証された本実装の値ではないため「見積もり」とします。
4. 以前の ChatGPT 会話は要求・仮説の回収に使い、事実の最終根拠にはしません。

## 2. ローカル根拠

| 対象 | 確認内容 |
|---|---|
| [`README.md`](../README.md) | 既存 SVS 入力、harmonic/noise pseudo mel、1-step、作者 RTF、多話者/style/OpenUTAU、license boundary |
| [`README.en.md`](../README.en.md) | 英語版の同一説明と SVC 入口 |
| [`LICENSE`](../LICENSE) | LeapSinger code の MIT license |
| [`leapsinger/models/svc.py`](../leapsinger/models/svc.py) | SVC model の実装境界 |
| [`leapsinger/modules/encoders/content_adapter.py`](../leapsinger/modules/encoders/content_adapter.py) | SVC condition adapter |
| [`svc_dataset.py`](../svc_dataset.py) | shard contract と validation |
| [`train.py`](../train.py) | architecture selection、学習・評価 wiring |
| [`infer.py`](../infer.py) | SVC single-item inference |
| [`configs/svc_base.yaml`](../configs/svc_base.yaml) | SVC baseline config |
| [`configs/3speaker_gan2d.yaml`](../configs/3speaker_gan2d.yaml) | 既存 SVS batch/update/GAN 設定の比較 |
| [`test_svc_model.py`](../test_svc_model.py) | targeted verification の範囲 |
| [`preprocess/svc/`](../preprocess/svc/) | 整列・部分集合・loudness・検査・coverage・split・report の実装 |
| [`test_svc_preprocess.py`](../test_svc_preprocess.py) / [`test_svc_dataset.py`](../test_svc_dataset.py) | 契約テスト（合計 366 件のうち 178 件） |
| [`configs/svc_base_multi.yaml`](../configs/svc_base_multi.yaml) / [`tools/m3_corpus.py`](../tools/m3_corpus.py) | M3 の recipe と素材の用意（話者ごとに 1 shard） |
| [`tools/m2_verify.py`](../tools/m2_verify.py) / [`tools/m3_verify.py`](../tools/m3_verify.py) / [`tools/nhv_indist.py`](../tools/nhv_indist.py) | M2 / M3 / M0 ゴール 4 の測定と、その JSON 報告 |
| [`tools/svc_convert.py`](../tools/svc_convert.py) / [`tools/audio_metrics.py`](../tools/audio_metrics.py) | 任意 WAV の変換 CLI と、帯域・spectral centroid の測定 |
| `log/m3_base/{config.yaml,perf.json,events}` と `out/m3_*/m3_report.json` | M3 の実験記録（回収済み。スループット・peak VRAM・内容保持の一次データ） |
| `.m0data/m4/m4_out/{commit.txt,config_used.yaml,perf.json,uv.lock,nvidia-smi.txt,train.log,events}` | M4 の実験記録（回収済み・sha256 照合済み）。**config の完全コピーと commit を含む** |
| `.m0data/m4/m4_out/{verify_*,self_*}` | M4 の測定結果。未知 source 40 clip × 5 checkpoint と、target hold-out の自己再構成 |
| `.m0data/m4/ckpt_{005000,010000,020000}.pt` | M4 の checkpoint。**採用は `ckpt_010000`**（選択規則は `configs/svc_target_ft.yaml` の header） |
| [`configs/svc_target_ft.yaml`](../configs/svc_target_ft.yaml) | M4 の recipe と、実験前に書いた checkpoint 選択規則 |
| [`tools/speaker_similarity.py`](../tools/speaker_similarity.py) / [`tools/speaker_calibrate.py`](../tools/speaker_calibrate.py) | 話者類似度の枠組みと、**歌声での較正**（module docstring に較正表。合格条件は事前登録） |
| `out/calib_{xvector,xvector_12s,ecapa,ecapa_12s,ecapa_20s}.json` | 較正の一次データ（encoder 2 本 × クリップ長 3 通り、VocalSet 20 歌手） |
| `out/sim/report_{base,ft10000,ft20000}.json` と `out/sim/breakdown.json` | M4 ゴール 2 の話者類似度（未知 source 6 clip、上限・下限つき、性別内訳） |
| [`tools/timing_metrics.py`](../tools/timing_metrics.py) / [`tools/asr_cer.py`](../tools/asr_cer.py) / [`tools/signal_quality.py`](../tools/signal_quality.py) / [`tools/rtf.py`](../tools/rtf.py) | M5 の客観指標 4 つ。**各 docstring に実測の限界**（分解能・言語依存・話し声モデル・段別 RTF） |
| `out/{rtf_cpu,cer_ja,cer_ft10000,sq_ft10000,timing_ft10000}.json` | 上記 4 指標の動作確認の一次データ（**測定本番ではない**） |
| [`test_svc_metrics.py`](../test_svc_metrics.py) | M5 の客観指標と道具の契約テスト 131 件 |
| `out/m5/record.json` | **M5 の実験記録**（両系の checkpoint / 設定 / 素材 / 全指標 / 判定と、その限界） |
| `out/m5/testset.json` | M5 の test set（seed から決定的。有声率と移調量つき） |
| `out/m5/metrics_{leapsvc,seedvc}/*.json` | 両系の測定（pitch / timing / similarity / cer / signal_quality / failures / rtf） |
| `out/m5/guard_rail.json` | 事前登録した判定の結果（**話者類似度で落ちた**） |
| `out/m5/blind/` | blind preference の材料（26 ペア、system 名を隠したもの）。**未実施** |
| `out/m5/probe{1..6}_*.json` | **話者性が弱い原因の調査**（層ごとの上限 / 漏れ / 話者条件 / mel の細部 / step 依存） |
| [`doc/svc-content-encoder.md`](svc-content-encoder.md) / [`doc/svc-dataset-ledger.md`](svc-dataset-ledger.md) | encoder 選定と M0 台帳 |
| [`pyproject.toml`](../pyproject.toml) / [`uv.lock`](../uv.lock) / `.python-version` | Python 3.13 固定、CUDA wheel index、依存の解決結果 |
| [`CLAUDE.md`](../CLAUDE.md) | コマンド、共有スタック、データ契約、既知の落とし穴 |
| [`doc/svc-plan.md`](svc-plan.md) | M0〜M6 の実行計画と完了条件 |

## 3. 公式リポジトリ

- [wavtechyukky repositories](https://github.com/wavtechyukky?tab=repositories) — 作者の現在の公開 repository 一覧。
- [LeapSinger](https://github.com/wavtechyukky/LeapSinger) — architecture、features、RTF、license note。
- [NHVSing](https://github.com/wavtechyukky/NHVSing) — vocoder variants、mel format、code / weight license note。
- [pyshiro](https://github.com/wavtechyukky/pyshiro) — HSMM/HMM alignment と GPL-3.0。
- [realigned-singing-labels](https://github.com/wavtechyukky/realigned-singing-labels) — label 内容と音声非同梱。
- [pitch-benchmark](https://github.com/wavtechyukky/pitch-benchmark) — pitch extractor comparison。
- [wavtechyukky/LeapSinger](https://github.com/wavtechyukky/LeapSinger) — **fork 元**。
- [GTSinger](https://github.com/AaronZ345/GTSinger) — 9 言語・技法ラベル付き歌唱コーパス（CC BY-NC-SA 4.0）。
- [VocalSet](https://zenodo.org/records/1442513) / [Annotated-VocalSet](https://zenodo.org/records/7061507) — 20 歌手・技法 17 種（CC BY 4.0）。
- [RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC) — rectified-flow SVC。ContentVec + RMVPE + 44.1 kHz + -18 LUFS。
- [Seed-VC](https://github.com/Plachtaa/seed-vc) — SVC baseline、model variants、steps、GPL-3.0、archive state。
- [RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC) — rectified-flow SVC の公開実装例。
- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) — differentiable-DSP SVC の公開実装例。

## 4. 論文

- Qian et al., [ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers](https://proceedings.mlr.press/v162/qian22b.html), ICML 2022.
- Zhou et al., [Simple and Effective Content Encoder for Singing Voice Conversion via Dimension Reduction](https://www.isca-archive.org/interspeech_2025/zhou25e_interspeech.pdf), Interspeech 2025 — content encoder を差し替えた同一 baseline 比較。次元削減の効果と離散トークンの out-of-domain 崩壊。
- [The Singing Voice Conversion Challenge 2025](https://arxiv.org/pdf/2509.15629) — 参加システムの表現選択。
- Wei et al., [RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music](https://www.isca-archive.org/interspeech_2023/wei23b_interspeech.html), Interspeech 2023.
- Plachta et al., [Seed-VC: High-Quality Zero-shot Voice Conversion and Singing Voice Conversion](https://arxiv.org/abs/2411.09943), 2024.
- [DAFMSVC](https://arxiv.org/abs/2508.05978) — flow-matching SVC の先行例。
- [Poly-SVC](https://arxiv.org/abs/2605.12310) — polyphonic SVC と harmonic modelling の先行例。
- [Singing Voice Conversion with Disentangled Representations of Singer and Vocal Technique Using Variational Autoencoders](https://arxiv.org/abs/2201.10130) — source-filter / harmonic signal を使う SVC の関連例。
- Liu et al., [The Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.html), Interspeech 2020.

## 5. 調査から導いたもの

次は一次資料の引用値ではなく、このプロジェクト向けの判断です。

- target singer 5〜10 時間を最初の推奨範囲とすること。
- multi-singer 20〜50 人 / 100〜300 時間を現実的な base 案とすること。
- 16 GB を target fine-tune、24 GB を base pretraining の基準とすること。
- offline teacher を先に作り、quality gate 後に streaming student を蒸留すること。
- Seed-VC を code dependency ではなく外部 baseline とすること。

これらは [データセットと計算資源](svc-data-compute.md)、[学習計画](svc-training.md)、[評価計画](svc-evaluation.md) の実験で更新します。

## 6. 更新ルール

- web source は URL とアクセス日を残す。
- GitHub の branch head ではなく、再現実験時は commit hash / release tag を固定する。
- model weight の license は repository code license から推測しない。
- benchmark 数値は dataset、metric、pre/postprocess を併記する。
- 実装変更時は [実装状況](svc-implementation-status.md) と索引の完了レベルを同時に更新する。
