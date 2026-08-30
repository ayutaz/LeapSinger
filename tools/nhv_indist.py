#!/usr/bin/env python3
"""同梱ボコーダー NHVSing にとって、各コーパスが in-distribution かを測る。

    uv run python tools/nhv_indist.py --out .m0data/nhv_indist --per-set 12

[実行計画](../doc/svc-plan.md) **M0 ゴール 4** は「base corpus が NHVSing にとって
in-distribution かを判断してある」ことを求め、その判断材料を M2 に先送りしていました
（[台帳](../doc/svc-dataset-ledger.md) 7 節）。ここがその実施です。

**測り方:** 音響モデルを介さず、**ground-truth mel をそのまま NHVSing に通した再合成**の
忠実度を比べます。音響モデルを挟むと、劣化が vocoder 由来か音響モデル由来か分かりません。
NHVSing の学習に使われたコーパス（波音リツ・夏目悠李・御丹宮くるみ・VocalSet）を対照に、
使われていない GTSinger を並べます。

**交絡に注意:** 収録品質そのものの差と、vocoder にとっての未知さは混ざり得ます。だから
対照を 2 種類に分けています — 同じ UTAU 系 DB（リツ・夏目・くるみ）と、**別コーパスだが
NHVSing が見ている** VocalSet です。VocalSet が GTSinger と同程度なら差はコーパスの性質、
VocalSet がリツ側に近いなら差は vocoder の既知/未知に帰属できます。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leapsinger.config import MelSpec  # noqa: E402
from leapsinger.mel import wav_to_mel_nhv  # noqa: E402

# NHVSing V3 の学習データ（doc/svc-dataset-ledger.md 7 節）に含まれるか。
DEFAULT_SETS = [
    ("ritsu",    "download/ritsu",              True,  "波音リツ（3 音源のうち normal）"),
    ("natsume",  "download/natsume",            True,  "夏目悠李"),
    ("oniku",    "download/oniku",              True,  "御丹宮くるみ"),
    ("vocalset", ".m0data/vocalset_wav",        True,  "VocalSet（別コーパスだが NHVSing 既知）"),
    ("gtsinger", ".m0data/gtsinger_ja/Japanese", False, "GTSinger 日本語（NHVSing 未学習）"),
]


def load_mono(path: Path, sr: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    wav, src_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if int(src_sr) != int(sr):
        from math import gcd
        g = gcd(int(src_sr), int(sr))
        wav = resample_poly(wav, sr // g, int(src_sr) // g).astype(np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32)


def peak_norm(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.abs(wav).max())
    return (wav / m * peak).astype(np.float32) if m > 1e-6 else wav


def pick_window(wav: np.ndarray, sr: int, seconds: float, f0_extract, hop: int,
                min_voiced: float = 0.5):
    """有声率が `min_voiced` 以上の窓を 1 つ返す。無ければ最も有声な窓。

    曲の先頭は無音やイントロのことが多く（M2 で 39% の chunk が完全に無声だった）、
    そのまま切ると vocoder の忠実度ではなく無音の再現を測ることになります。
    """
    n = int(seconds * sr)
    if len(wav) < n:
        return None
    best = None
    for frac in (0.35, 0.5, 0.65, 0.2, 0.8):
        a = int((len(wav) - n) * frac)
        seg = peak_norm(np.ascontiguousarray(wav[a:a + n]))
        f0, uv = f0_extract(seg, sr, hop)
        ratio = float(np.mean(np.asarray(uv) > 0.5))
        if best is None or ratio > best[0]:
            best = (ratio, seg, np.asarray(f0, np.float32), np.asarray(uv, np.float32))
        if ratio >= min_voiced:
            break
    return best


def collect(root: Path, want: int, seed: int) -> list[Path]:
    """曲（親ディレクトリ）が偏らないように、決定的に候補を並べる。

    短すぎる clip や無声だけの clip は後段で落ちるので、**必要数より多めに**返します。
    """
    wavs = sorted(p for p in root.rglob("*.wav"))
    if not wavs:
        return []
    by_group: dict[str, list[Path]] = {}
    for p in wavs:
        by_group.setdefault(str(p.parent), []).append(p)
    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    order = [groups[i] for i in rng.permutation(len(groups))]
    out, i = [], 0
    while len(out) < want and i < want * 8:
        g = order[i % len(order)]
        files = by_group[g]
        idx = i // len(order)
        if idx < len(files):
            out.append(files[idx])
        i += 1
    return out[:want]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".m0data/nhv_indist")
    ap.add_argument("--per-set", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-voiced", type=float, default=0.3,
                    help="有声率がこれ未満の clip は測らない（前処理本体と同じ扱い）")
    ap.add_argument("--device", default="cuda", help="RMVPE の device（推論のみ）")
    ap.add_argument("--vocoder", default="checkpoints/nhv_v3.onnx")
    ap.add_argument("--save-wav", type=int, default=2, help="各セットで残す試聴用 WAV の数")
    a = ap.parse_args()

    import soundfile as sf

    from infer import load_vocoder, mel_to_wav
    from preprocess.f0_rmvpe import extract_f0_rmvpe

    mel_spec = MelSpec()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocoder = load_vocoder(str(ROOT / a.vocoder), sr=mel_spec.sr)

    def f0_extract(wav, sr, hop):
        return extract_f0_rmvpe(wav, int(sr), int(hop), 65.0, 1100.0,
                                device=a.device, interpolate=False)

    def mel_of(wav):
        return wav_to_mel_nhv(wav, sr=mel_spec.sr, n_fft=mel_spec.n_fft, hop=mel_spec.hop,
                              win=mel_spec.win, n_mels=mel_spec.n_mels,
                              fmin=mel_spec.fmin, fmax=mel_spec.fmax)

    report: dict = {"mel": mel_spec.to_dict(), "seconds": a.seconds, "per_set": a.per_set,
                    "seed": a.seed, "vocoder": a.vocoder, "sets": {}}
    t0 = time.time()

    for name, rel, in_nhv, note in DEFAULT_SETS:
        root = ROOT / rel
        if not root.exists():
            print(f"[skip] {name}: {rel} が無い", flush=True)
            report["sets"][name] = {"skipped": f"{rel} が無い", "in_nhvsing": in_nhv}
            continue
        files = collect(root, a.per_set * 3, a.seed)
        rows, saved, too_short, unvoiced = [], 0, 0, 0
        for path in files:
            if len(rows) >= a.per_set:
                break
            wav = load_mono(path, mel_spec.sr)
            got = pick_window(wav, mel_spec.sr, a.seconds, f0_extract, mel_spec.hop,
                              min_voiced=a.min_voiced)
            if got is None:
                too_short += 1
                continue
            voiced_ratio, seg, f0, uv = got
            if voiced_ratio < a.min_voiced:
                # 前処理本体と同じ扱い。無声だけの clip を測ると vocoder の忠実度ではなく
                # 無音の再現を測ることになる（M2 で実際に踏んだ欠陥）。
                unvoiced += 1
                continue
            mel = mel_of(seg)
            hat = mel_to_wav(vocoder, mel, np.log2(np.maximum(f0, 1.0)), uv)
            k = min(len(seg), len(hat))
            mel_hat = mel_of(np.ascontiguousarray(hat[:k]))
            t = min(mel.shape[1], mel_hat.shape[1])
            d = np.abs(mel[:, :t] - mel_hat[:, :t])

            f0_hat, uv_hat = f0_extract(np.ascontiguousarray(hat[:k]), mel_spec.sr, mel_spec.hop)
            f0_hat = np.asarray(f0_hat, np.float32)[:t]
            uv_hat = np.asarray(uv_hat, np.float32)[:t]
            both = (uv[:t] > 0.5) & (uv_hat > 0.5) & (f0[:t] > 0) & (f0_hat > 0)
            row = {
                "file": str(path.relative_to(ROOT)),
                "voiced_ratio": voiced_ratio,
                "mel_l1": float(d.mean()),
                "mel_l1_p95": float(np.percentile(d, 95)),
                "uv_agree": float(np.mean((uv[:t] > 0.5) == (uv_hat > 0.5))),
            }
            if int(both.sum()) >= 16:
                cents = 1200.0 * np.log2(f0_hat[both] / f0[:t][both])
                row["f0_median_semitones"] = float(np.median(np.abs(cents)) / 100.0)
                row["f0_corr"] = float(np.corrcoef(np.log2(f0[:t][both]),
                                                   np.log2(f0_hat[both]))[0, 1])
            rows.append(row)
            if saved < a.save_wav:
                stem = f"{name}_{saved}"
                sf.write(out_dir / f"{stem}_orig.wav", seg, mel_spec.sr)
                sf.write(out_dir / f"{stem}_resynth.wav", hat[:k], mel_spec.sr)
                saved += 1
            print(f"  {name:<9} {path.name[:38]:<38} melL1 {row['mel_l1']:.4f} "
                  f"voiced {voiced_ratio:.2f}", flush=True)

        def agg(key, rows=rows):        # rows を明示的に束縛（ループ変数の捕捉を避ける）
            vals = [r[key] for r in rows if key in r]
            return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                    "std": float(np.std(vals)), "n": len(vals)} if vals else None

        report["sets"][name] = {
            "note": note, "in_nhvsing": in_nhv, "root": rel, "n_clips": len(rows),
            "rejected_too_short": too_short, "rejected_unvoiced": unvoiced,
            "mel_l1": agg("mel_l1"), "mel_l1_p95": agg("mel_l1_p95"),
            "uv_agree": agg("uv_agree"), "f0_median_semitones": agg("f0_median_semitones"),
            "f0_corr": agg("f0_corr"), "voiced_ratio": agg("voiced_ratio"),
            "clips": rows,
        }
        print(f"[{name}] {len(rows)} clips  mel L1 "
              f"{report['sets'][name]['mel_l1']['mean']:.4f}", flush=True)

    report["elapsed_sec"] = round(time.time() - t0, 1)
    (out_dir / "nhv_indist.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== NHVSing 再合成の忠実度（ground-truth mel 経由。低いほど良い） ===")
    print(f"{'set':<10}{'NHV 学習':<9}{'clips':>6}{'mel L1':>9}{'中央値':>9}{'p95':>8}"
          f"{'F0 半音':>9}{'V/UV':>8}{'除外':>7}")
    for name, s in report["sets"].items():
        if s.get("skipped"):
            print(f"{name:<10}{'-':<9}{'skip':>6}  {s['skipped']}")
            continue
        f0m = s["f0_median_semitones"]
        drop = s["rejected_too_short"] + s["rejected_unvoiced"]
        print(f"{name:<10}{'あり' if s['in_nhvsing'] else 'なし':<9}{s['n_clips']:>6}"
              f"{s['mel_l1']['mean']:>9.4f}{s['mel_l1']['median']:>9.4f}"
              f"{s['mel_l1_p95']['mean']:>8.3f}"
              f"{(f0m['median'] if f0m else float('nan')):>9.3f}"
              f"{s['uv_agree']['mean']:>8.3f}{drop:>7}")
    print(f"\n-> {out_dir / 'nhv_indist.json'}  ({report['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
