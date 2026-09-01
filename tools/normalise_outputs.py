#!/usr/bin/env python3
"""外部 baseline の出力を、このリポジトリの測定ツールが読める形へ揃える（M5）。

    uv run python tools/normalise_outputs.py --seedvc out/m5/seedvc \
      --segments out/m5/seedvc/_segments --out out/m5/seedvc_norm

Seed-VC は `<tag>/vc_<tag>__<ref>_<設定>.wav` に書きますが、[`timing_metrics.py`](timing_metrics.py)
などは `<name>_converted.wav` / `<name>_source.wav` の組を読みます。**測定ツールを baseline に
合わせて書き換えるのではなく、入力の形だけを揃えます**（測定側を触ると、これまでの結果と
比較できなくなります）。

**source は変換に渡した区間そのものを置きます。** timing と CER は source を基準にするので、
別の区間を source として置くと比較が壊れます。ここでは `--segments`（Seed-VC に渡した
切り出し済み WAV）をそのまま `_source.wav` にします。

**上限（`*_vocoder_only.wav`）は作りません。** 「GT mel を自分のボコーダーに通した再合成」は
系ごとに違う量なので、外部 baseline に対して同じものを作れません
（[評価計画](../doc/svc-evaluation.md) 4 節）。したがって baseline 側で測れるのは、
**上限を要らない指標**（timing・話者類似度）だけです。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def normalised_name(tag: str, kind: str) -> str:
    """測定ツールが読む形のファイル名。"""
    if kind not in ("converted", "source"):
        raise ValueError(f"kind は converted / source（{kind!r} が来ました）")
    return f"{tag}_{kind}.wav"


def find_seedvc_outputs(root: Path) -> dict[str, Path]:
    """`<tag>/vc_*.wav` を tag -> path で返す。

    **1 つの tag に出力が 2 つあるときは例外にします。** 設定違いの出力が残っていると、
    どちらを測ったのか分からなくなります。
    """
    out: dict[str, Path] = {}
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if d.name.startswith("_"):
            continue
        wavs = sorted(d.glob("*.wav"))
        if not wavs:
            continue
        if len(wavs) > 1:
            raise ValueError(
                f"{d} に出力が {len(wavs)} 本あります（{[w.name for w in wavs]}）。"
                "設定違いの出力が混ざっていると、どちらを測ったのか分からなくなります")
        out[d.name] = wavs[0]
    return out


def main() -> int:
    import argparse
    import json
    import shutil

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seedvc", required=True, help="Seed-VC の出力の根")
    ap.add_argument("--segments", required=True,
                    help="変換に渡した切り出し済み WAV（`<tag>.wav`）の置き場")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    found = find_seedvc_outputs(Path(a.seedvc))
    seg_root = Path(a.segments)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing = []
    for tag, conv in sorted(found.items()):
        src = seg_root / f"{tag}.wav"
        if not src.exists():
            missing.append(tag)
            continue
        shutil.copyfile(conv, out / normalised_name(tag, "converted"))
        shutil.copyfile(src, out / normalised_name(tag, "source"))
        rows.append({"tag": tag, "converted_from": str(conv), "source_from": str(src)})

    print(f"[normalise] {len(rows)} clip -> {out}")
    if missing:
        print(f"  ** source が見つからない tag: {missing} **")
    (out / "normalise_manifest.json").write_text(
        json.dumps({"n": len(rows), "missing_source": missing, "clips": rows,
                    "note": ("上限は作っていない。系が違うので「GT mel を自分のボコーダーに"
                             "通した再合成」は共有できない。上限が要る指標は測れない")},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
