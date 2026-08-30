# SVC content encoder の選定

調査日: 2026-08-30

対象ブランチ: `feature/svc`

[SVC アーキテクチャ](svc-architecture.md) 3 節の「**未決:** 採用モデル、層、sampling rate、stride、
正規化、mel grid への resampling 方法」を埋めるための調査。ライセンスと次元は Hugging Face の
model API で実際に確認した値を記載する。

## 1. 結論

**決定:** content encoder は **ContentVec**（`lengyue233/content-vec-best`、MIT、768 次元、layer 12）
を凍結して使う。抽出は 2 段に分け、**1 段目の cache に生の 768 次元**を、**`svc_shard.npz` には
そこから固定のランダムに選んだ 256 次元**を書く（7 節）。mel grid への整列は **left（直前保持）**、
F0 は **RMVPE 固定**、loudness 特徴量は **dataset 統計で正規化**する。

**根拠:** 下の 3 節（Interspeech 2025 の同一 test set 比較）で、ContentVec を 256 次元へ削減したものが
生の 768 次元を主観類似度で 0.53 ポイント上回っている。とくに **out-of-domain（学習言語以外）での
差が大きく、日本語歌唱に英語・中国語で学習した SSL を当てる本プロジェクトの状況に対応する。**

## 2. 候補一覧

**確認済み**（ライセンス・アーキテクチャは Hugging Face model API の応答。次元は各モデルの公称値）:

| 候補 | 次元 | ライセンス | 分類 | 備考 |
|---|---:|---|---|---|
| **ContentVec** `lengyue233/content-vec-best` | 768 | **MIT** | 話者情報を明示的に除去 | so-vits-svc / RVC / RIFT-SVC の標準。DL 249k |
| HuBERT-base `facebook/hubert-base-ls960` | 768 | Apache-2.0 | 素の SSL | ContentVec の土台。話者情報が残る |
| HuBERT-large `facebook/hubert-large-ll60k` | 1024 | Apache-2.0 | 素の SSL | |
| WavLM-base+ / large `microsoft/wavlm-*` | 768 / 1024 | **宣言なし（要確認）** | 素の SSL | 雑音に強いが話者情報を保持する設計 |
| XLS-R 300M `facebook/wav2vec2-xls-r-300m` | 1024 | Apache-2.0 | 多言語 SSL | Seed-VC の realtime 版が採用 |
| w2v-BERT 2.0 `facebook/w2v-bert-2.0` | 1024 | MIT | 多言語 SSL | SeamlessM4T 由来。DL 2.1M |
| chinese-hubert-large `TencentGameMate/...` | 1024 | MIT | 素の SSL | so-vits の `cnhubertlarge` |
| **日本語 HuBERT + 音素 CTC** `prj-beatrice/japanese-hubert-base-phoneme-ctc-v2` | 768 | Apache-2.0 | 音素 CTC | CTC が音素以外を捨てる＝別ルートで話者情報が落ちる |
| 日本語 HuBERT `reazon-research/japanese-hubert-base-k2` | 768 | Apache-2.0 | 素の SSL | disentangle なし |
| HuBERT-soft `bshall/hubert` | 256 | 要確認 | soft unit | Soft-VC。離散化の情報欠落を緩和 |
| Whisper encoder `openai/whisper-*` | 768〜1280 | Apache-2.0 | ASR/PPG | so-vits の `whisper-ppg`。**30 秒窓・重い** |
| MERT `m-a-p/MERT-v1-*` | 768 / 1024 | **CC-BY-NC-4.0** | 音楽 SSL | **非商用。Release 配布と衝突するため不採用** |
| XEUS | 1024 | 要確認 | 多言語 SSL | 4,000 言語。HF から直接取得できず未確認 |
| ASTRAL-Quantization | — | 要確認 | 話者分離トークナイザ | Seed-VC V2 が採用 |
| 離散トークン（k-means） | — | — | 離散 | **out-of-domain で崩壊（3 節）** |

`rinna/japanese-hubert-base` は Hugging Face の model API で見つからなかった（削除か改名の可能性）。
実在を確認できないため候補に含めない。

## 3. 決め手になった実証データ

**確認済み（一次資料）:** Zhou et al., *Simple and Effective Content Encoder for Singing Voice
Conversion via Dimension Reduction*, Interspeech 2025, pp. 1268–1272
（[ISCA archive](https://www.isca-archive.org/interspeech_2025/zhou25e_interspeech.pdf)）。
so-vits-svc を土台にした同一 baseline・同一データで content encoder だけを差し替えた比較。
F0 は RMVPE、singer embedding は WeSpeaker ResNet34。

**Other Languages（学習言語以外＝ out-of-domain。本プロジェクトに対応する条件）の結果:**

| content encoder | 次元 | SSIM(WeSpeaker)↑ | SSIM(CAM++)↑ | CMOS（内容）↑ | SMOS（類似度）↑ |
|---|---:|---:|---:|---:|---:|
| SSL-Token（k-means 10,000） | 768 | 0.837 | 0.641 | **3.210** | 4.006 |
| SSL-Emb（素の SSL） | 768 | 0.755 | 0.462 | 4.326 | **2.494** |
| SSL-Soft-Emb | 256 | 0.774 | 0.484 | 4.266 | 3.154 |
| ContentVec-Emb | 768 | 0.799 | 0.554 | 4.250 | 3.512 |
| SSL-128-Emb | 128 | 0.821 | 0.591 | 4.220 | 3.726 |
| **ContentVec-256-Emb** | **256** | **0.835** | **0.634** | 4.256 | **4.038** |

読み取れること:

1. **素の SSL は timbre 漏れが大きい。** SMOS 2.494 は他の全条件より 1 ポイント以上低い。
   [SVC アーキテクチャ](svc-architecture.md) がリスクとして挙げた現象が数字で出ている。
2. **ContentVec への切り替えで SMOS 2.494 → 3.512。** 話者分離学習の効果。
3. **さらに 256 次元へ削減して 3.512 → 4.038。** 内容の正確さ（CMOS）はほぼ不変（4.250 → 4.256）。
4. **離散トークンは out-of-domain で内容が壊れる。** CMOS 3.210 は全条件で最低。
   in-domain（中国語）では 4.086 なので、**学習言語外での崩壊**という性質。

手法は「768 次元からランダムに部分集合を選び、学習中は固定する」だけで、実装コストはほぼゼロ。
**選んだ index を manifest に記録すれば再現できる。**

## 4. 先行実装が実際に採用しているもの

**確認済み（各リポジトリの README）:**

| 実装 | content encoder | F0 | 備考 |
|---|---|---|---|
| [RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC) | **ContentVec** | RMVPE | rectified flow SVC＝本プロジェクトに最も近い。44,100 Hz、**loudness を -18 LUFS へ正規化**。**V3 で Whisper encoder を削除**した |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) | 歌声: Whisper-small / realtime: XLSR-large / V2: ASTRAL-Quantization | 記載なし | 比較対象。歌声モデル `seed-uvit-whisper-base` は 44,100 Hz |
| [so-vits-svc 4.1](https://github.com/svc-develop-team/so-vits-svc) | `vec768l12`（ContentVec）ほか `vec256l9` / `hubertsoft` / `whisper-ppg` / `cnhubertlarge` / `dphubert` / `wavlmbase+` | 選択式 | 既定は ContentVec。whisper-ppg は 30 秒未満の制約あり |

RIFT-SVC が V2 → V3 で Whisper encoder を落としている点は、**構成が最も近い実装が Whisper を
割に合わないと判断した**という意味で参考になる。加えて Whisper の 30 秒窓は
[実行計画](svc-plan.md) M6 の streaming student と相性が悪い。

## 5. 採用しない理由

| 候補 | 理由 |
|---|---|
| MERT | **CC-BY-NC-4.0。** 非商用限定であり、学習済みモデルを Release で配布する形と衝突する |
| 離散トークン全般 | out-of-domain で内容が壊れる（3 節）。日本語歌唱＝ SSL の学習分布外なので直撃する |
| Whisper / whisper-ppg | 30 秒窓、計算量、RIFT-SVC が V3 で削除。M6 の streaming に不利 |
| WavLM | ライセンスが HF のカードで宣言されておらず未確認。話者情報を保持する設計で層選択がシビア |
| 素の HuBERT | timbre 漏れが最大（SMOS 2.494）。**ただし ablation のベースラインとしては有用** |

## 6. 補間方法と部分集合の決定

**決定（実測にもとづく）:** 補間は **left（直前保持・左寄せ繰り返し）**。

SSL は 16 kHz・stride 320 = 50 Hz、mel grid は 44,100/256 = 172.265625 Hz、比 3.4453125 は整数に
なりません。3 方式を同条件で測った結果:

| 方式 | left 基準の先読み | ブレンドされるフレーム |
|---|---:|---:|
| **left** | **0 ms** | **0%** |
| nearest | 20 ms | 0% |
| linear | 20 ms | 99.4% |

left は他方式に対して劣る点がありません。nearest と linear は 1 SSL フレーム（20 ms）先を読むため、
[実行計画](svc-plan.md) M6 の streaming で無償の lookahead を積み、train と inference の因果性も
ずれます。linear はさらに 99.4% のフレームを混ぜ、ContentVec の表現空間に実在しないベクトルを作ります。
so-vits-svc の `repeat_expand_2d` の既定 `mode='left'` とも一致します。

**決定:** 256 次元の部分集合は **seed 0 を既定**とします。
Interspeech 2025 の 2 つの部分集合は SSIM 0.813 / 0.822 と小さいながら差がありました。
**未実施:** seed 1 との比較。M2 では実施せず、**M3 の本学習の前**に 1 度だけ確かめます
（shard は両方あり、2 段目の再実行だけで作れます）。

**未決:** layer 12 は so-vits-svc の `vec768l12` に合わせた既定値だが、
`content-vec-best` の final projection 出力との対応は実装時に実際の shape で確認する。

## 7. 実装への含意

- 抽出器は **encoder 非依存**に作る。モデル名・層・stride・正規化を引数にし、manifest へ記録する。
  比較（[実行計画](svc-plan.md) M3 / M5 の ablation）が後から回せることを優先する。
- **抽出を 2 段に分ける。** 1 段目（重い）が WAV から 50 Hz の生 ContentVec 768 と F0 を cache に出し、
  2 段目（軽い）が mel grid への整列・正規化・256 次元の切り出しを行って `svc_shard.npz` を書く。
  補間方法と 256 次元 seed の ablation が **2 段目の再実行だけ**で済み、ContentVec と RMVPE を
  回し直さずに比較できる。shard に書くのは 256 次元（loader に切り出させると契約の性質を崩すため）。
- ただし `configs/svc_base.yaml` の `content_dim` は既定 256 とする。
  **見積もり:** 768 → 256 で content の VRAM とディスクが約 1/3 になり、
  vast.ai の時間課金とストレージ課金の両方に効く。
- 音声側の loudness 正規化は RIFT-SVC に倣って **-18 LUFS** を検討する。これは
  「loudness 特徴量を dataset 統計で標準化する」という決定とは別レイヤの話で、両立する。
