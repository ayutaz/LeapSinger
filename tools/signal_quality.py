#!/usr/bin/env python3
"""信号品質を測る（[実行計画](../doc/svc-plan.md) M5 ゴール 2）。

    uv run python tools/signal_quality.py --dir out/sim/ft10000 --device cpu --out out/sq.json

**歌声への妥当性は未検証です。** 使える採点器（torchaudio の SQUIM、DNSMOS 系）はどれも
**話し声で学習**されています。歌唱は基本周波数の範囲も持続音の割合も違うので、**絶対値を
「音質」として主張しません。**

そのかわり、このリポジトリの他の指標と同じく**上限**（GT mel をボコーダーに通した再合成）と
並べ、**そこからの差**を読みます。上限は「この mel 表現とこのボコーダーで到達できる上限」なので、
差のぶんだけが音響モデルに帰せられます。**上限が無ければ数値を出しません。**

同梱の `SquimScorer` は `torchaudio.pipelines.SQUIM_OBJECTIVE`（参照不要で STOI / PESQ /
SI-SDR を推定）を使います。**新しい依存は増えません**（torchaudio は speechbrain と一緒に
入っています）。`score` を差し替えれば DNSMOS などにできます。
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

Score = Callable[[np.ndarray, int], dict[str, float]]

CAVEAT = ("採点器は話し声で学習されている。歌唱への妥当性は未検証なので、"
          "絶対値ではなく上限（GT mel の再合成）からの差だけを読むこと")


def quality_report(converted: np.ndarray, *, ceiling: np.ndarray | None, sr: int,
                   score: Score) -> dict[str, Any]:
    """変換後と上限を採点し、**差**を返す。

    片側にしか無い指標は `unpaired` へ入れ、`gap` に混ぜません（差を 0 と書くと、
    測れていないことが見えなくなります）。
    """
    if ceiling is None:
        raise ValueError(
            "上限（GT mel をボコーダーに通した再合成）が要ります。採点器は話し声で学習されて"
            "いるので、絶対値には意味がありません（svc_convert.py --self-check で作れます）")

    cnv = score(converted, sr)
    ceil = score(ceiling, sr)
    shared = sorted(set(cnv) & set(ceil))
    return {
        "converted": dict(cnv),
        "ceiling": dict(ceil),
        "gap": {k: float(cnv[k]) - float(ceil[k]) for k in shared},
        "unpaired": sorted(set(cnv) ^ set(ceil)),
        "caveat": CAVEAT,
    }


class SquimScorer:
    """torchaudio SQUIM。16 kHz の波形 -> {stoi, pesq, si_sdr}（参照不要の推定値）。

    **推定値であって実測ではありません。** 参照信号を使わずに STOI / PESQ / SI-SDR を
    予測するモデルです。歌唱への妥当性は未検証（module の docstring）。
    """

    def __init__(self, device: str = "cpu"):
        import torch
        from torchaudio.pipelines import SQUIM_OBJECTIVE
        self.device = device
        self.bundle = SQUIM_OBJECTIVE
        self.model = SQUIM_OBJECTIVE.get_model().to(device).eval()
        self._torch = torch

    def __call__(self, wav: np.ndarray, sr: int) -> dict[str, float]:
        if int(sr) != int(self.bundle.sample_rate):
            raise ValueError(f"{self.bundle.sample_rate} Hz を渡すこと（{sr} が来ました）")
        x = self._torch.from_numpy(
            np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            stoi, pesq, si_sdr = self.model(x)
        return {"stoi": float(stoi.item()), "pesq": float(pesq.item()),
                "si_sdr": float(si_sdr.item())}

    def manifest(self) -> dict[str, Any]:
        return {"scorer": "torchaudio SQUIM_OBJECTIVE", "scorer_sr": int(self.bundle.sample_rate),
                "scorer_caveat": CAVEAT}


def main() -> int:
    import argparse
    import json

    import soundfile as sf

    from preprocess.svc.extract import _resample

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="`*_converted.wav` と上限を含むディレクトリ")
    ap.add_argument("--ceiling-suffix", default="_vocoder_only.wav",
                    help="上限（GT mel 再合成）の接尾辞。svc_convert.py --self-check が作る")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.dir)
    scorer = SquimScorer(device=a.device)
    target_sr = int(scorer.bundle.sample_rate)

    def load(p):
        w, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        return _resample(np.ascontiguousarray(w, dtype=np.float32), sr, target_sr)

    pairs = []
    for cnv in sorted(root.glob("*_converted.wav")):
        stem = cnv.name[: -len("_converted.wav")]
        ceil = root / f"{stem}{a.ceiling_suffix}"
        if ceil.exists():
            pairs.append((stem, cnv, ceil))
    if not pairs:
        sys.exit(f"{a.dir} に converted と上限（{a.ceiling_suffix}）の組がありません。"
                 "上限は svc_convert.py --self-check で作れます")

    rows = []
    for stem, cnv, ceil in pairs:
        r = quality_report(load(cnv), ceiling=load(ceil), sr=target_sr, score=scorer)
        r["file"] = stem
        rows.append(r)
        g = r["gap"]
        print(f"  {stem:40s} " + "  ".join(f"{k} {g[k]:+.3f}" for k in sorted(g)))

    keys = sorted({k for r in rows for k in r["gap"]})
    summary = {"n_clips": len(rows),
               "gap_median": {k: float(np.median([r["gap"][k] for r in rows if k in r["gap"]]))
                              for k in keys},
               "converted_median": {k: float(np.median([r["converted"][k] for r in rows]))
                                    for k in keys},
               "ceiling_median": {k: float(np.median([r["ceiling"][k] for r in rows]))
                                  for k in keys}}
    print("\n=== 信号品質（上限との差の中央値）===")
    for k in keys:
        print(f"  {k:8s} 変換 {summary['converted_median'][k]:7.3f}  "
              f"上限 {summary['ceiling_median'][k]:7.3f}  差 {summary['gap_median'][k]:+7.3f}")
    print(f"\n  ** {CAVEAT} **")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"summary": summary, "manifest": scorer.manifest(), "clips": rows},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
