#!/usr/bin/env python3
"""M5 の test set を決定的に組む（[実行計画](../doc/svc-plan.md) M5「決定 4」）。

    uv run python tools/m5_testset.py --unseen-root .m0data/vocalset_calib \
      --target-root download/ritsu/DATABASE --out out/m5/testset.json

**手で選ぶと偏ります。** M3 の測り直しで、`rglob` の並び順のまま 20 clip を取ったら
**18 女性 / 2 男性**になり、「男性 source が暗い」という結論を一度取り逃がしました。
事前登録した構成をコードで満たさせます。

| 条件 | 値 | 理由 |
|---|---|---|
| 未知 source | **20 clip**、**1 歌手 1 clip** | 1 人の癖が test set を支配しないように。VocalSet は 20 名なので全員が 1 本ずつ入る |
| 性別 | **各性別が最低 30%** | 回復率の伸びが倍違う（同性 +17.4 点 / 異性 +8.4 点）。ちょうど半々は課さない（素材が 女 9 / 男 11 のため） |
| target hold-out | **6 clip** | ゴール 1 が held-out song も要求する |
| 長さ | **12 秒以上** | 話者類似度の較正がその長さでしか通らない |
| 移調 | **男性 source は +12 半音**、女性と hold-out は 0 | 出力の傾斜が入力 F0 に強く従う |

**ちょうど半々にしないのは素材の制約です。** VocalSet は女性 9 名・男性 11 名なので、
「男女 10 本ずつ・1 歌手 1 clip」は組めません。**1 歌手 1 clip を優先**し、性別は下限
（各 30%）だけを課します。M3 で 18 女性 / 2 男性になった失敗は、この下限で止まります。

**選択は seed から決定的**で、manifest に seed と規則を残します。後から「なぜこの 20 本か」を
復元できないと、比較の土台が消えます。
"""
from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# 話者類似度の較正が通る最短の長さ（tools/speaker_similarity.py の MIN_SECONDS と同じ）。
MIN_SECONDS = 12.0

# 男性 source -> 女性 target の既定の移調量（doc/svc-plan.md M4 の前提）。
MALE_TRANSPOSE = 12


# 各性別が test set に占める最低割合。M3 では 18 女性 / 2 男性（男性 10%）で
# 「男性 source が暗い」を取り逃がした。
MIN_GENDER_SHARE = 0.30


def select_clips(pool: Sequence[dict[str, Any]], *, n: int, seed: int,
                 min_seconds: float = MIN_SECONDS,
                 min_gender_share: float = MIN_GENDER_SHARE) -> list[dict[str, Any]]:
    """**1 歌手 1 clip**で `n` 本選び、各性別が `min_gender_share` を下回らないことを保証する。

    **ちょうど半々にはしません。** 素材の歌手数が性別で違うため（VocalSet は 女 9 / 男 11）、
    半々を課すと 1 歌手から 2 本取ることになり、その歌手の癖が test set に効きます。
    **歌手の多様性を優先**し、性別は下限だけを見ます。

    **足りないなら黙って偏らせず例外にします。** 偏った set を返すと、それに気づかないまま
    比較してしまいます（M3 で実際に起きました）。
    """
    usable = [c for c in pool if float(c.get("seconds", 0.0)) >= min_seconds]
    if not usable:
        raise ValueError(
            f"{min_seconds} 秒以上の clip がありません（話者類似度の較正がその長さでしか"
            "通らないので、短い clip を入れると測れなくなります）")

    by_spk: dict[str, list[dict[str, Any]]] = {}
    for c in usable:
        by_spk.setdefault(str(c["speaker"]), []).append(c)
    if len(by_spk) < n:
        raise ValueError(
            f"歌手が {len(by_spk)} 名しかおらず、{n} 本を 1 歌手 1 clip で選べません"
            "（1 人から複数取ると、その歌手の癖が test set を支配します）")

    rng = random.Random(seed)
    speakers = sorted(by_spk)
    rng.shuffle(speakers)
    picked = speakers[:n]

    counts: dict[str, int] = {}
    for spk in picked:
        counts[str(by_spk[spk][0].get("gender"))] = counts.get(
            str(by_spk[spk][0].get("gender")), 0) + 1
    floor = int(n * min_gender_share)
    for gender in ("female", "male"):
        if counts.get(gender, 0) < floor:
            raise ValueError(
                f"{gender} が {counts.get(gender, 0)} 本しかなく、下限 {floor} 本"
                f"（{min_gender_share:.0%}）を満たしません。**偏った test set を黙って返しません**"
                "（M3 で 18 女性 / 2 男性になり、男性 source の問題を取り逃がしました）")

    out: list[dict[str, Any]] = []
    for spk in picked:
        clips = sorted(by_spk[spk], key=lambda c: str(c.get("clip", "")))
        out.append(dict(clips[rng.randrange(len(clips))]))
    return out


def segment_holdout(songs: Sequence[dict[str, Any]], *, n: int, seconds: float,
                    seed: int) -> list[dict[str, Any]]:
    """hold-out 曲から**重ならない**区間を `n` 本切り出す。

    hold-out は 3 曲しかありませんが 1 曲 3〜5 分あります。**曲単位 split を保ったまま**
    区間を分けて本数を作ります（別の曲を混ぜて水増ししません）。

    - 各曲へ本数を均等に割り当てる
    - **区間は重ねない**（同じところを 2 回測ると n を水増ししただけになる）
    - **曲全体へ散らす**（曲頭に固まるとイントロばかりになり、無声率が高くて検証に向かない）
    """
    if not songs:
        raise ValueError("hold-out 曲がありません")
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    order = sorted(songs, key=lambda c: str(c.get("song", "")))
    for i, song in enumerate(order):
        want = n // len(order) + (1 if i < n % len(order) else 0)
        if want <= 0:
            continue
        full = float(song.get("full_seconds", 0.0))
        if full < want * seconds:
            raise ValueError(
                f"曲 {song.get('song')!r} は {full:.1f} 秒しかなく、{seconds} 秒 x {want} 本を"
                "重ねずに取れません（区間を重ねると n を水増ししただけになります）")
        # 曲を want 個の窓へ等分し、各窓の中でずらす。**窓をまたがないので重ならない。**
        span = full / want
        for k in range(want):
            slack = max(0.0, span - seconds)
            start = k * span + rng.uniform(0.0, slack)
            c = dict(song)
            c.update({"start": float(start), "seconds": float(seconds),
                      "clip": f"{song.get('song')}_{k}", "transpose": 0})
            out.append(c)
    return out


def build_testset(unseen: Sequence[dict[str, Any]], holdout: Sequence[dict[str, Any]], *,
                  n_unseen: int, n_holdout: int, seed: int,
                  min_seconds: float = MIN_SECONDS,
                  clip_seconds: float | None = None) -> dict[str, Any]:
    """未知 source と target hold-out を分けて選び、移調量まで決めて返す。

    **移調を運用者に任せません。** 男性 source を移調し忘れると、モデルではなく source と
    target の音域差を測ることになります。
    """
    picked = select_clips(unseen, n=n_unseen, seed=seed, min_seconds=min_seconds)
    for c in picked:
        c["transpose"] = MALE_TRANSPOSE if c.get("gender") == "male" else 0

    usable_hold = [c for c in holdout if float(c.get("seconds", 0.0)) >= min_seconds]
    if len(usable_hold) >= n_holdout:
        rng = random.Random(seed + 1)
        hold = [dict(c) for c in rng.sample(
            sorted(usable_hold, key=lambda c: str(c.get("clip"))), n_holdout)]
    else:
        # 曲数が足りないときは、**曲単位 split を保ったまま**区間へ分ける。
        hold = segment_holdout(usable_hold, n=n_holdout,
                               seconds=float(clip_seconds or min_seconds), seed=seed + 1)
    for c in hold:
        # target 自身の曲は音域が合っている。移調すると別の実験になる。
        c["transpose"] = 0

    return {
        "unseen": picked,
        "holdout": hold,
        "manifest": {
            "seed": int(seed), "n_unseen": int(n_unseen), "n_holdout": int(n_holdout),
            "min_seconds": float(min_seconds), "male_transpose": MALE_TRANSPOSE,
            "min_gender_share": MIN_GENDER_SHARE,
            "rule": ("1 歌手 1 clip・12 秒以上・各性別が最低 30%。男性 source のみ +12 半音。"
                     "hold-out は移調しない"),
        },
    }


def _scan_vocalset(root: Path, glob: str) -> list[dict[str, Any]]:
    """VocalSet 形式（`<speaker>/excerpts/straight/*.wav`）を pool にする。"""
    import soundfile as sf

    from tools.speaker_calibrate import gender_of
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for w in sorted(d.glob(glob)):
            out.append({"speaker": d.name, "gender": gender_of(d.name), "clip": w.name,
                        "path": str(w), "seconds": float(sf.info(w).duration),
                        "kind": "unseen"})
    return out


def _scan_target(root: Path, songs: Sequence[str], seconds: float) -> list[dict[str, Any]]:
    """target の hold-out 曲だけを pool にする。**学習に使った曲を混ぜないこと。**"""
    import soundfile as sf

    out = []
    want = {s for s in songs}
    for w in sorted(root.rglob("*.wav")):
        song = w.parent.name
        if song not in want:
            continue
        dur = float(sf.info(w).duration)
        if dur < seconds:
            continue
        out.append({"speaker": "ritsu", "gender": "female", "clip": w.name, "path": str(w),
                    "song": song, "seconds": seconds, "full_seconds": dur, "kind": "holdout"})
    return out


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unseen-root", required=True,
                    help="未知 source の根（VocalSet 形式の話者ディレクトリ）")
    ap.add_argument("--unseen-glob", default="excerpts/straight/*.wav")
    ap.add_argument("--target-root", required=True, help="target の WAV の根")
    ap.add_argument("--holdout-song", action="append", default=None,
                    help="hold-out 曲名（複数可）。学習の split と一致させること")
    ap.add_argument("--n-unseen", type=int, default=20)
    ap.add_argument("--n-holdout", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=20.0, help="各 clip から切り出す長さ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not a.holdout_song:
        raise SystemExit(
            "--holdout-song を指定してください。**学習の split と一致していないと leakage します。**\n"
            "M4（eval_songs=3 / seed 42）なら: anywhere-3_normal / boukyakumoyou-3_normal / "
            "skyhighblue-3_normal")

    unseen = _scan_vocalset(Path(a.unseen_root), a.unseen_glob)
    holdout = _scan_target(Path(a.target_root), a.holdout_song, a.seconds)
    print(f"[testset] pool: 未知 source {len(unseen)} clip / hold-out {len(holdout)} clip")

    ts = build_testset(unseen, holdout, n_unseen=a.n_unseen, n_holdout=a.n_holdout,
                       seed=a.seed, clip_seconds=a.seconds)
    ts["manifest"]["unseen_root"] = str(a.unseen_root)
    ts["manifest"]["unseen_glob"] = a.unseen_glob
    ts["manifest"]["target_root"] = str(a.target_root)
    ts["manifest"]["holdout_songs"] = list(a.holdout_song)
    ts["manifest"]["seconds"] = a.seconds

    n_m = sum(1 for c in ts["unseen"] if c["gender"] == "male")
    print(f"  未知 source {len(ts['unseen'])} clip（男 {n_m} / 女 {len(ts['unseen']) - n_m}）"
          f"、うち男性は +{MALE_TRANSPOSE} 半音")
    print(f"  hold-out    {len(ts['holdout'])} clip（移調なし）")
    for c in ts["unseen"][:4]:
        print(f"    {c['speaker']:9s} {c['clip']:28s} tp={c['transpose']:+d}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(ts, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
