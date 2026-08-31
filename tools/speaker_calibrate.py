#!/usr/bin/env python3
"""話者照合 encoder が「歌声で使えるか」を較正する（M4 ゴール 2 / M5 ゴール 2 の前提）。

    uv run python tools/speaker_calibrate.py --root .m0data/vocalset_wav \
      --encoder ecapa --clips-per-speaker 3 --out out/calib_ecapa.json

**target similarity を報告する前に、必ずこれを通します。** 話者照合の埋め込みは同一話者でも
1.0 にならず、別話者でも 0 になりません。**絶対値ではなく 3 群の分離**で判定します。

| 群 | 意味 |
|---|---|
| **同一話者** | 別クリップどうし。上限の分布 |
| **別話者・同性** | **ここと重なると target similarity を主張できない** |
| 別話者・異性 | 最も分かれやすい。ここだけで判定しない |

判定は**重なり**（同一話者ペアの下位 5% を超える別話者・同性ペアの割合）で行います。平均どうしを
比べるとばらつきの大きい encoder を通してしまうので、分布の裾で見ます。

**閾値は encoder を走らせる前に決めます。** 走らせてから決めると、出た数字に合わせて基準の
ほうを動かせてしまいます（M4 の checkpoint 選択規則と同じ理由）。**このリポジトリの事前登録値は
`MAX_OVERLAP = 0.20`** です。

**確認済み（2026-09-01）:** `transformers` の x-vector 2 本はこの較正を通らず（6 秒 83.3% /
12 秒 77.0%）、**ECAPA-TDNN を 12 秒以上**で使うと通ります（19.8% / 20 秒 17.3%）。
[`speaker_similarity.py`](speaker_similarity.py) の docstring に較正表。

### 素材の用意（VocalSet、再現手順）

較正の結論は素材の条件に依存します。**同一技法・別の曲の抜粋**（`excerpts/straight`、
1 歌手 3 曲）が最も公平で、`arpeggios`（同じ音階を単母音で歌う）は話者性の手がかりが乏しく
不利になります。VocalSet の配布 zip から次のように取り出します。

    uv run python -c "
    import zipfile, pathlib
    out = pathlib.Path('.m0data/vocalset_calib')
    with zipfile.ZipFile('.m0data/vocalset_audio.zip') as z:
        for n in z.namelist():
            if n.lower().endswith('.wav') and '/excerpts/straight/' in n:
                parts = n.split('/')          # FULL/<speaker>/excerpts/straight/<clip>.wav
                d = out / parts[1] / 'excerpts' / 'straight'
                d.mkdir(parents=True, exist_ok=True)
                (d / parts[-1]).write_bytes(z.read(n))
    "

20 歌手（female 9 / male 11）× 3 本 = 60 clip になります。**下限に使う「無関係な話者」は、
target と同じ系統の DB から取ること**（録音条件の違いを話者性と取り違えないため。M4 では
波音リツに対し鬼灯・棗を使いました）。
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# **事前登録した合格条件。** 実験の後に動かさないこと。
MAX_OVERLAP = 0.20

_GENDERS = ("female", "male")


def gender_of(name: str) -> str:
    """話者ディレクトリ名から性別を読む（VocalSet の `female1` / `male10`）。

    **推測しません。** 取り違えると「異性なら分かれる」という結論ごと壊れるので、
    知らない命名は例外にします。
    """
    low = str(name).strip().lower()
    for g in _GENDERS:
        if low.startswith(g):
            return g
    raise ValueError(
        f"話者名 {name!r} から性別を読めません。VocalSet の female<N> / male<N> 形式か、"
        "--gender-map で明示してください")


def pair_groups(speakers: Sequence[str],
                genders: Sequence[str] | None = None) -> dict[str, list[tuple[int, int]]]:
    """クリップの添字ペアを 3 群へ分ける。**自分自身との組は作りません。**"""
    spk = list(speakers)
    gen = list(genders) if genders is not None else [gender_of(s) for s in spk]
    if len(gen) != len(spk):
        raise ValueError(f"speakers {len(spk)} 本に対し genders {len(gen)} 本です")
    out: dict[str, list[tuple[int, int]]] = {
        "same": [], "diff_same_gender": [], "diff_cross_gender": []}
    for i in range(len(spk)):
        for j in range(i + 1, len(spk)):
            if spk[i] == spk[j]:
                out["same"].append((i, j))
            elif gen[i] == gen[j]:
                out["diff_same_gender"].append((i, j))
            else:
                out["diff_cross_gender"].append((i, j))
    return out


def overlap_fraction(same: Sequence[float], diff: Sequence[float], *, pct: float = 5.0) -> float:
    """同一話者ペアの下位 `pct`% を超える別話者ペアの割合。

    **平均どうしの差では判定しません。** 平均が離れていても分布が重なっていれば、
    個々のクリップについて何も言えないためです。
    """
    if not len(same) or not len(diff):
        raise ValueError(f"空の群では重なりを測れません（same {len(same)} / diff {len(diff)}）")
    thr = float(np.percentile(np.asarray(same, dtype=np.float64), pct))
    return float(np.mean(np.asarray(diff, dtype=np.float64) >= thr))


def _agg(vals: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std()), "median": float(np.median(a)),
            "n": int(a.size)}


def calibration_report(embeddings: Sequence[np.ndarray], speakers: Sequence[str],
                       genders: Sequence[str] | None = None) -> dict[str, Any]:
    """3 群の cos 分布と重なりを返す。**上限か最難条件が作れないなら例外にします。**"""
    from tools.speaker_similarity import cosine

    if len(embeddings) != len(speakers):
        raise ValueError(f"埋め込み {len(embeddings)} 本に対し話者 {len(speakers)} 本です")
    groups = pair_groups(speakers, genders)
    if not groups["same"]:
        raise ValueError(
            "同一話者の別クリップペアが 0 組です。上限が作れないので判定できません"
            "（1 話者あたり 2 クリップ以上を渡すこと）")
    if not groups["diff_same_gender"]:
        raise ValueError(
            "別話者・同性のペアが 0 組です。この encoder の最も厳しい条件を測っていません"
            "（同性の話者を 2 名以上入れること）")

    vals = {k: [cosine(embeddings[i], embeddings[j]) for i, j in pairs]
            for k, pairs in groups.items()}
    rep: dict[str, Any] = {k: _agg(v) for k, v in vals.items() if v}
    for k in groups:
        rep.setdefault(k, {"mean": float("nan"), "sd": float("nan"),
                           "median": float("nan"), "n": 0})
    rep["overlap"] = overlap_fraction(vals["same"], vals["diff_same_gender"])
    if vals["diff_cross_gender"]:
        rep["overlap_cross_gender"] = overlap_fraction(vals["same"], vals["diff_cross_gender"])
    rep["n_clips"] = int(len(speakers))
    rep["n_speakers"] = int(len(set(speakers)))
    return rep


def passes_calibration(report: dict[str, Any], *, max_overlap: float = MAX_OVERLAP) -> bool:
    """重なりが事前登録した閾値以下か。**判定に使った閾値を報告へ書き残します。**"""
    passed = bool(report["overlap"] <= max_overlap)
    report["verdict"] = {"passed": passed, "max_overlap": float(max_overlap),
                         "overlap": float(report["overlap"]),
                         "criterion": "同一話者ペアの下位 5% を超える別話者・同性ペアの割合"}
    return passed


def _collect(root: Path, clips_per_speaker: int, seconds: float, sr: int,
             glob: str = "**/*.wav") -> tuple[list[np.ndarray], list[str]]:
    """話者ディレクトリごとに同じ本数だけ読む。**本数を揃えないと群の重みが偏ります。**

    `glob` は話者ディレクトリからの相対パターンです。**「どのクリップで較正したか」は
    結論そのもの**なので、歌唱と朗読、技法違いを取り違えないよう明示的に絞ります
    （VocalSet なら `excerpts/straight/*.wav` が最も公平な条件）。

    同一話者ペアを作れない話者（1 本しか無い）は**黙って落とします**。上限が作れないためです。
    """
    from tools.speaker_similarity import _load

    wavs: list[np.ndarray] = []
    speakers: list[str] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        paths = sorted(d.glob(glob))[:clips_per_speaker]
        loaded = _load(paths, sr, seconds)
        if len(loaded) < 2:
            continue
        wavs.extend(loaded)
        speakers.extend([d.name] * len(loaded))
    return wavs, speakers


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="話者ごとのディレクトリを含む WAV の根")
    ap.add_argument("--encoder", default="ecapa", choices=("ecapa", "xvector"))
    ap.add_argument("--model", default=None, help="encoder の model id を上書きする")
    ap.add_argument("--clips-per-speaker", type=int, default=3)
    ap.add_argument("--glob", default="**/*.wav",
                    help="話者ディレクトリからの相対パターン（VocalSet: excerpts/straight/*.wav）")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="各クリップの長さ。**6 秒では通らない**（12 秒から通る）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-overlap", type=float, default=MAX_OVERLAP,
                    help=f"合格とする重なりの上限（事前登録値 {MAX_OVERLAP}）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from tools.speaker_similarity import EcapaEncoder, XVectorEncoder
    if a.encoder == "ecapa":
        enc = EcapaEncoder(a.model or "speechbrain/spkrec-ecapa-voxceleb", device=a.device)
    else:
        enc = XVectorEncoder(a.model or "microsoft/wavlm-base-plus-sv", device=a.device)

    wavs, speakers = _collect(Path(a.root), a.clips_per_speaker, a.seconds, 16000, a.glob)
    if not wavs:
        sys.exit(f"{a.root} から WAV を読めませんでした")
    print(f"[calib] {len(wavs)} clips / {len(set(speakers))} speakers  encoder={a.encoder}")
    embs = [enc(w, 16000) for w in wavs]
    rep = calibration_report(embs, speakers)
    ok = passes_calibration(rep, max_overlap=a.max_overlap)
    rep["manifest"] = enc.manifest() | {
        "root": str(a.root), "glob": a.glob, "clips_per_speaker": a.clips_per_speaker,
        "seconds": a.seconds}

    print("=== 較正 ===")
    for k, label in (("same", "同一話者      "), ("diff_same_gender", "別話者・同性  "),
                     ("diff_cross_gender", "別話者・異性  ")):
        g = rep[k]
        print(f"  {label}: {g['mean']:.4f} ± {g['sd']:.4f}  (n={g['n']})")
    print(f"  重なり（同性）: {rep['overlap'] * 100:.1f}%   閾値 {a.max_overlap * 100:.0f}%")
    print(f"  -> {'合格。同性間の target similarity を報告してよい' if ok else '不合格。この encoder では同性間の target similarity を報告しない'}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
