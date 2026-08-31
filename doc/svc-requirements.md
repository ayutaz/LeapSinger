# LeapSVC 要件

## 1. 目的

**決定:** 特定の target singer へ追加学習でき、未知の source singer の歌唱内容・音程・強弱を保ちながら、target singer の音色で再合成する SVC を作ります。

最終的な狙いはリアルタイム利用ですが、最初に高品質な offline teacher を完成させ、比較評価を通過してから streaming student を作ります。

## 2. 必須要件

- source WAV から content、F0、V/UV、loudness を抽出できること。
- 歌詞、音素ラベル、MusicXML、MIDI、source/target の平行録音を必須にしないこと。
- source の歌詞内容、音程曲線、発声タイミングをできるだけ保持すること。
- target singer の音色を再現すること。
- 既存 LeapSinger の SVS 経路を壊さないこと。
- NHVSing へ渡せる 44.1 kHz・hop 256・128-bin ln-mel を生成すること。
- 学習、評価、推論が同じ特徴量定義を使い、frame 数の不一致を暗黙補正しないこと。
- base model と target fine-tune の成果物を別ディレクトリに保存し、base を上書きしないこと。

## 3. 望ましい要件

- target singer 5〜10 時間程度から実用品質を狙えること。これは実測前の目標値です。
- 1-step flow の速度上の利点を維持すること。
- source singer が学習話者に含まれなくても変換できること。
- 強声、弱声、裏声、息成分、ビブラート、ロングトーンを不自然に均さないこと。
- offline teacher の品質を大きく落とさず causal または limited-lookahead 化できること。
- 推論時に speaker ID を要求しない target 固定モデルへ変換できること。

## 4. 非目標

- zero-shot で任意の短い参照音声から新しい target voice を生成する汎用 VC。
- 歌詞から歌唱を生成する text-to-singing 製品の置き換え。
- speech VC を含む万能変換器。
- 現段階での「Seed-VC より高品質」「世界初」「リアルタイム対応」という断定。

## 5. 入出力契約

### 学習入力

各 phrase に対し、同じフレーム長 `T` を持つ次の配列を用意します。

| 特徴量 | shape | 意味 |
|---|---:|---|
| content | `[T, content_dim]` | 凍結した SSL encoder の表現 |
| f0_interp | `[T]` | 無声 gap を補間した Hz 単位の F0 |
| uv | `[T]` | 有声 1、無声 0 |
| loudness | `[T]` | 全データで定義を統一した log-RMS 等 |
| mel | `[128, T]` | target の NHVSing 互換 ln-mel |

### 推論入力・出力

```text
source WAV
  -> content / f0_interp / uv / loudness
  -> target-conditioned mel
  -> NHVSing(mel, source F0)
  -> converted WAV
```

## 6. 学習モデルの関係

```text
凍結 SSL content encoder
            |
multi-singer nonparallel singing corpus
            |
      offline SVC base
            |
 target singer fine-tune
            |
   target-specific teacher
            |
   distillation / causality
            |
     streaming student
```

**推奨:** target singer の自己再構成だけで始める PoC は許容しますが、最終品質では multi-singer base を経由します。単一話者だけでは SSL feature に残った話者性へ依存し、未知 source singer への変換で崩れる危険が高いためです。

## 7. 品質ゲート

### Gate A: データ

- 収録・利用・学習・生成物配布の権利を確認済み。
- clipping、伴奏漏れ、強い reverb、誤った sample rate を検査済み。
- 音域と発声スタイルの coverage を可視化済み。
- train / validation / test を曲単位・録音セッション単位で分離済み。

**確認済み（2026-08-30）: Gate A は入手できた素材について通過しました。** 5 コーパスに検査・
coverage・split を実データで通し、権利条件を規約 URL と取得日つきで記録しています
（[データセット台帳](svc-dataset-ledger.md)、[実行計画](svc-plan.md) M0）。**未取得の東北きりたん /
No.7 は対象外**で、発声スタイルの coverage は技法ラベルを持つコーパスでのみ自動集計できます。

### Gate B: offline

- held-out song と未知 source singer で変換できる。
- target similarity、F0、明瞭度、ノイズの客観指標を記録する。
- **target similarity については、使う speaker encoder が歌声で較正（上限・下限・重なり）を
  通っていること。** 2026-08-31 時点で手元の x-vector 2 本は同性の歌手を分離できず、
  **この項目は測定手段から未達**です（[評価計画](svc-evaluation.md) 4 節）。
- **未知 source の内容保持が base から落ちていないこと。** M4 で、target の再現と未知 source の
  保持が単調に逆へ動くことを実測しました（[実行計画](svc-plan.md) M4）。
- Seed-VC と blind listening test を実施する。
- failure sample を除外せず分類して残す。

### Gate C: streaming

- Gate B を満たした teacher を基準に student を評価する。
- chunk 境界に click、音切れ、F0 jump がない。
- RTF、algorithmic lookahead、audio I/O を含む end-to-end latency を実機測定する。
- 長時間連続運転で buffer overrun / underrun がない。

## 8. 要ユーザー判断

**決着済み（2026-08-30）:**

| 項目 | 決定 |
|---|---|
| 用途と配布 | **研究・個人利用のみ、モデルの配布なし**。CC-NC 系や research-only の素材も使える |
| 生成音声の商用利用 | **行わない** |
| target singer と追加 corpus の権利条件 | 波音リツ（target）/ GTSinger + 日本語 3 DB（base）/ VocalSet（未知 source test）。権利条件は [データセット台帳](svc-dataset-ledger.md) |

**未決:**

- 音質と遅延のどちらを優先するか。
- blind test で合格とする最小差、評価者数、対象楽曲。
- **将来 配布や商用利用へ移る場合**、使える素材が大きく変わります（台帳 5 節の B / C）。
  その時点で素材の選定からやり直す必要があります。
