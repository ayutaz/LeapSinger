# LeapSVC 学習計画

## 1. 推奨する段階

```text
Phase 0  前処理・小規模 overfit smoke
Phase 1  multi-singer offline base pretraining
Phase 2  target singer fine-tune
Phase 3  optional NHVSing target fine-tune
Phase 4  offline quality gate
Phase 5  streaming student distillation
```

品質比較前に Phase 0〜3 の実行記録を残し、Phase 4 を通過するまで Phase 5 のリアルタイム最適化を主目的にしません。

## 2. Phase 0: 前処理と overfit smoke

1. 権利確認済みの短い isolated vocal から shard を作る。
2. 1〜数 phrase を train とし、loss が十分低下するまで overfit する。
3. mel と WAV を保存し、F0 timing、無声区間、長さを確認する。
4. 同じ入力・seed・checkpoint で再実行し、再現性を確認する。
5. train と infer の特徴量正規化が同一であることを検証する。

この段階の成功は一般化や音質を示しません。loader、condition、flow、vocoder の実配線を検証する smoke です。

## 3. Phase 1: multi-singer base

**推奨:** 20〜50 人、合計 100〜300 時間を最初の本学習案とし、より小さい corpus で先に recipe を確定します。

```text
Singer A WAV -> content/F0/loudness + speaker A -> mel A
Singer B WAV -> content/F0/loudness + speaker B -> mel B
Singer C WAV -> content/F0/loudness + speaker C -> mel C
```

speaker-balanced sampling を使い、長時間話者だけが batch を支配しないようにします。性別・音域・言語の比率だけでなく、録音条件の偏りも監視します。

初期 loss:

- rectified-flow loss
- mel reconstruction loss
- 必要に応じて V/UV / spectral auxiliary loss

GAN は baseline が安定し、artifact の比較が可能になった後に別 run で導入します。

## 4. Phase 2: target fine-tune

1. base checkpoint を immutable input として保存する。
2. target 専用の run directory を作る。
3. 低い learning rate と短い validation interval から始める。
4. target similarity と unseen-source intelligibility の両方を見る。
5. train reconstruction だけが改善し、未知 source が悪化する場合は早期停止する。

target singer だけでゼロから学習する PoC も可能ですが、SSL content 内の target timbre をそのまま利用する shortcut を学ぶ危険があります。そのため最終モデルでは base pretraining を推奨します。

## 5. 既存 SVS checkpoint の warm-start

### 再利用しやすい部分

- harmonic/noise excitation の設定と実装
- rectified-flow backbone
- mel reconstruction / flow loss
- optional discriminator
- NHVSing interface

### 直接再利用できない部分

- phoneme embedding
- duration に基づく length regulation
- phoneme encoder の入力 projection

### 提案する手順

1. SVS flow と excitation を読み込む。
2. SSL encoder、flow、vocoder を一時的に凍結する。
3. ContentAdapter が旧 phoneme condition と互換な hidden 表現を出すよう、利用可能なら condition distillation を行う。
4. flow を解凍し、SVC reconstruction / flow loss で共同学習する。
5. 独立初期化 baseline と比較する。

**未実装:** 現在の `--init_from --finetune` は model 全体を同一構造として読み込む前提であり、SVS から SVC へ安全に部分ロードする機能ではありません。部分 weight mapping、missing/unexpected key の allowlist、ロード結果の記録とテストが必要です。

## 6. Phase 3: NHVSing fine-tune

NHVSing の既存重みを固定した baseline を先に作ります。次の場合に限り target fine-tune を比較します。

- target singer の高音や裏声で vocoder artifact が系統的に出る。
- ground-truth mel を入力しても同じ artifact が出る。
- acoustic model の誤差と vocoder の誤差を切り分けられる。

vocoder fine-tune は target similarity を上げる可能性がありますが、mel 分布への過適合、データ規約、重み配布条件を別に確認します。

## 7. Phase 4: offline quality gate

[評価計画](svc-evaluation.md) に従い、同一 source、同一 target、同一音量処理で baseline を比較します。held-out song と未知 source singer を必須とします。最良 checkpoint を train loss だけで選びません。

## 8. Phase 5: streaming distillation

**未実装:** teacher の full-context 出力を基準に、causal または limited-lookahead student を学習します。

候補:

- chunk-aware training と state cache
- teacher mel / velocity distillation
- boundary-aware loss
- random chunk length と left-context augmentation
- vocoder の overlap-add / stateful inference

streaming student は offline teacher と同じ test set で比較し、品質低下と遅延低下を同時に報告します。

## 9. 初期コマンド

現在の実装が受け取る target-singer 学習例です。`svc_shard.npz` はまだ repository 内 command では生成できません。

```powershell
uv run python -m train --config configs/svc_base.yaml `
  --data_dirs data/target `
  --run_name svc_target `
  --out_root log `
  --device cuda
```

multi-singer の場合は `model.n_speakers`、`data.spk_map`、`train.balance_speakers` を corpus に合わせます。

## 10. 実験記録

各 run で最低限保存します。

- Git commit / dirty diff、config の完全コピー
- dataset manifest / split / checksums
- dependency lock（`uv.lock` と `.python-version`）と CUDA / driver / GPU。現在の基準環境は [実装状況](svc-implementation-status.md) の「実行環境」表
- seed、AMP、batch frames、peak VRAM
- init checkpoint と load report
- train/validation curve、生成 sample、failure list
- best checkpoint の選択規則
- 途中再開の履歴

**決定:** 高い learning rate や overfit 継続は baseline checkpoint と出力先を分け、元の結果を上書きしません。
