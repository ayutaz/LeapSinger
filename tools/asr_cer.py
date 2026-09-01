#!/usr/bin/env python3
"""明瞭度を CER で測る（[実行計画](../doc/svc-plan.md) M5 ゴール 2）。

    uv run python tools/asr_cer.py --dir out/sim/ft10000 --device cpu --out out/cer.json

**歌唱の ASR は当てになりません。** 話し声用に学習された認識器を、伴奏なしとはいえ歌に当てて
います。したがって**「変換後の CER が X%」という単独の数値には意味がありません。** 必ず 2 つの
基準と並べます（[`m3_verify.py`](m3_verify.py) の content cos と同じ読み方）。

| 基準 | 意味 |
|---|---|
| **source** | 参照テキスト。歌詞は未知なので **source の書き起こしを参照**にする |
| **上限（ceiling）** | GT mel をボコーダーに通した再合成の CER。**ASR とボコーダー由来の誤りの下駄** |
| 変換 | 変換後の CER。上限との差が、音響モデルに帰せられる分 |

**上限が無ければ報告しません。** 上限を測らずに変換後の CER だけを出すと、ASR が歌を
聞き取れないぶんまで音響モデルのせいにすることになります。`--self-check` で上限の WAV を
作れます（[`svc_convert.py`](svc_convert.py)）。

**ASR が空を返すことは実際に起こります。** そのときは `asr_failed` を立て、CER を `None` に
します（0% と区別が付かなくなるため）。

**注意: これは「明瞭度」の代理であって歌詞の正解率ではありません。** 参照が人手の歌詞では
なく source の書き起こしなので、source の時点で誤認識された語はそのまま参照になります。

### 実測で分かった制約（2026-09-01）

**1. `--language` を素材に合わせること。** VocalSet はイタリア語（Caro mio ben）・英語
（Row your boat）・ラテン語（Dona nobis pacem）の楽曲です。`--language ja` を強制したところ
**source・変換・上限がすべて別物に書き起こされ、CER は 3 clip とも 1.0** になりました。

**2. 上限が判別不能なら差を出しません。** 上のとき上限も変換も 1.0 で、差は 0.0 でした。
**差だけ見れば「音響モデルは劣化させていない」と読めてしまいます。** 実際は判別できて
いないだけなので、上限 CER が `CEILING_MAX_CER` を超えたら `ceiling_unusable` を立て、
差を `None` にします。

**日本語素材では機能します。** 波音リツの自己再構成 1 clip で **上限 0.0 / 変換 0.60**
（差 +0.60）でした。上限が 0.0 ということは、ボコーダーと ASR は無罪で、**差はすべて
音響モデルに帰属**します。**n=1 なので幅は読まないこと。**
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

Transcribe = Callable[[Any], str]

# **上限の CER がこれを超えたら、この指標は判別できていません。** 実測（Whisper small を
# VocalSet の歌唱に当てた）で上限も変換も CER 1.0 になり、差が 0.0 になりました。差だけ見ると
# 「音響モデルは劣化させていない」と読めてしまうので、そのときは差を出しません。
CEILING_MAX_CER = 0.5

_DROP = re.compile(r"[\s、。，．,\.!！?？「」『』（）\(\)・…ー~〜\- - :：;；\"'`]")


def normalize_ja(text: str) -> str:
    """日本語の書き起こしを比較用に正規化する。

    NFKC で全角英数を半角へ、英字は小文字へ畳み、空白と句読点・記号を落とします。
    **カナは畳みません**（「シャツ」と「シヤツ」を同じにすると CER が甘くなります）。
    """
    s = unicodedata.normalize("NFKC", str(text))
    return _DROP.sub("", s).lower()


def cer(ref: str, hyp: str, *, normalize: bool = True) -> float:
    """文字誤り率 = レーベンシュタイン距離 / 参照の文字数。

    **1 で頭打ちにしません。** 幻聴で長く書き起こすと 1 を超えますが、そこを丸めると
    「全部間違い」と「参照より長く誤る」を区別できなくなります。
    """
    r = normalize_ja(ref) if normalize else str(ref)
    h = normalize_ja(hyp) if normalize else str(hyp)
    if not r:
        raise ValueError("参照が空です。CER を定義できません（ASR が失敗した可能性）")

    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[len(h)] / len(r)


def cer_report(source_wav: Any, converted_wav: Any, *, ceiling_wav: Any,
               transcribe: Transcribe) -> dict[str, Any]:
    """source を参照、上限を下駄として変換後の CER を返す。

    **上限（`ceiling_wav`）は必須です。** 無いと ASR とボコーダー由来の誤りを音響モデルに
    帰属させてしまいます。
    """
    if ceiling_wav is None:
        raise ValueError(
            "上限（GT mel をボコーダーに通した再合成）が要ります。変換後の CER だけでは、"
            "ASR とボコーダー由来の誤りを音響モデルのせいにしてしまいます "
            "（svc_convert.py --self-check で作れます）")

    # **1 クリップにつき 1 回だけ書き起こす。** ASR は重い。
    t_src = transcribe(source_wav)
    t_cnv = transcribe(converted_wav)
    t_ceil = transcribe(ceiling_wav)

    out: dict[str, Any] = {
        "transcripts": {"source": t_src, "converted": t_cnv, "ceiling": t_ceil},
        "asr_failed": not normalize_ja(t_src),
        "cer_converted": None, "cer_ceiling": None, "cer_excess_over_ceiling": None,
        "ceiling_unusable": False, "ceiling_max_cer": CEILING_MAX_CER,
    }
    if out["asr_failed"]:
        return out
    out["cer_converted"] = cer(t_src, t_cnv)
    out["cer_ceiling"] = cer(t_src, t_ceil)
    # **上限が既に判別不能なら差を出さない。** 上限も変換も全滅なら差は 0 になり、
    # 「音響モデルは劣化させていない」と読めてしまう。判別できていないだけ。
    out["ceiling_unusable"] = bool(out["cer_ceiling"] > CEILING_MAX_CER)
    if not out["ceiling_unusable"]:
        out["cer_excess_over_ceiling"] = out["cer_converted"] - out["cer_ceiling"]
    return out


class WhisperTranscriber:
    """Whisper による書き起こし。**新しい依存は増えません**（transformers は既に使っています）。

    **話し声で学習されたモデルを歌に当てている**点は、この道具の限界そのものです。だからこそ
    上限（GT mel の再合成）と並べて読みます。`--model` で差し替えられます。
    """

    def __init__(self, model_id: str = "openai/whisper-small", device: str = "cpu",
                 language: str = "ja"):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.model_id, self.device, self.language = model_id, device, language
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device).eval()
        self._torch = torch

    def __call__(self, wav_path: Any) -> str:
        import numpy as np
        import soundfile as sf

        from preprocess.svc.extract import _resample
        w, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        w = _resample(np.ascontiguousarray(w, dtype=np.float32), sr, 16000)
        x = self.proc(w, sampling_rate=16000, return_tensors="pt")
        with self._torch.no_grad():
            ids = self.model.generate(
                x.input_features.to(self.device), language=self.language, task="transcribe",
                max_new_tokens=200)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def manifest(self) -> dict[str, Any]:
        return {"asr": self.model_id, "asr_language": self.language, "asr_sr": 16000,
                "asr_caveat": "話し声で学習されたモデルを歌に当てている。上限と並べて読むこと"}


def main() -> int:
    import argparse
    import json

    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="`*_source.wav` / `*_converted.wav` / `*_vocoder_only.wav` を含むディレクトリ")
    ap.add_argument("--ceiling-suffix", default="_vocoder_only.wav",
                    help="上限（GT mel 再合成）の接尾辞。svc_convert.py --self-check が作る")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--language", default="ja",
                    help="**素材の言語に合わせること。** VocalSet はイタリア語・英語・"
                         "ラテン語の楽曲なので ja を強制すると全滅する（実測）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.dir)
    trios = []
    for src in sorted(root.glob("*_source.wav")):
        stem = src.name[: -len("_source.wav")]
        cnv = root / f"{stem}_converted.wav"
        ceil = root / f"{stem}{a.ceiling_suffix}"
        if cnv.exists() and ceil.exists():
            trios.append((stem, src, cnv, ceil))
    if not trios:
        sys.exit(f"{a.dir} に source / converted / 上限（{a.ceiling_suffix}）の 3 点組が"
                 "ありません。上限は svc_convert.py --self-check で作れます")

    tr = WhisperTranscriber(a.model, device=a.device, language=a.language)
    rows = []
    for stem, src, cnv, ceil in trios:
        r = cer_report(src, cnv, ceiling_wav=ceil, transcribe=tr)
        r["file"] = stem
        rows.append(r)
        if r["asr_failed"]:
            print(f"  {stem:40s} ASR が source を書き起こせず（この clip は除外）")
        elif r["ceiling_unusable"]:
            print(f"  {stem:40s} 上限 CER {r['cer_ceiling'] * 100:5.1f}% "
                  f"> {CEILING_MAX_CER * 100:.0f}% -> **判別不能**（差を出さない）")
        else:
            print(f"  {stem:40s} 変換 {r['cer_converted'] * 100:5.1f}%  "
                  f"上限 {r['cer_ceiling'] * 100:5.1f}%  "
                  f"差 {r['cer_excess_over_ceiling'] * 100:+5.1f} 点")

    ok = [r for r in rows if not r["asr_failed"] and not r["ceiling_unusable"]]
    summary = {
        "n_clips": len(rows), "n_usable": len(ok),
        "asr_failed": sum(1 for r in rows if r["asr_failed"]),
        "ceiling_unusable": sum(1 for r in rows if r["ceiling_unusable"]),
        "cer_converted": float(np.median([r["cer_converted"] for r in ok])) if ok else None,
        "cer_ceiling": float(np.median([r["cer_ceiling"] for r in ok])) if ok else None,
    }
    if ok:
        summary["cer_excess_over_ceiling"] = (summary["cer_converted"] - summary["cer_ceiling"])
    print("\n=== CER（中央値）===")
    if not ok:
        print(f"  **この素材とこの ASR では CER を測れません。**"
              f"（ASR 失敗 {summary['asr_failed']} / 上限が判別不能 "
              f"{summary['ceiling_unusable']} / 全 {len(rows)}）")
        print("  --language が素材の言語と一致しているか、モデルが歌唱を扱えるかを見直すこと")
    else:
        print(f"  変換 : {summary['cer_converted'] * 100:.1f}%")
        print(f"  上限 : {summary['cer_ceiling'] * 100:.1f}%（ASR とボコーダーの下駄）")
        print(f"  差   : {summary['cer_excess_over_ceiling'] * 100:+.1f} 点 <- 音響モデルの分")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"summary": summary, "manifest": tr.manifest(), "clips": rows},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
