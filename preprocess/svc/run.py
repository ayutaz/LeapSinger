"""WAV ディレクトリから `data/<db>/{metadata.json, svc_shard.npz, manifest.json}` を作る。

    uv run python -m preprocess.svc.run --wav-dir download/ritsu --out data/ritsu_svc
    uv run python -m preprocess.svc.run --wav-dir download/ritsu --out data/ritsu_svc --limit 8
    uv run python -m preprocess.svc.run --from-cache data/ritsu_svc/_cache --out data/ritsu_svc --subset-seed 1

抽出は 2 段です。1 段目（重い）が ContentVec と RMVPE を回して `_cache/` へ、2 段目（軽い）が
整列・正規化・次元削減を行って shard を書きます。**`--from-cache` で 2 段目だけを回せます**。
補間方法や 256 次元 seed の ablation は、ContentVec と RMVPE を回し直さずに済みます。

phrase 名は `{song}_{NNNN}` です。`dataset.py` の `_song_of()` が曲単位の train/eval 分割に
使うので、この命名を崩すと leakage 防止が効かなくなります。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from leapsinger.config import MelSpec

from .chunk import chunk_spans, voiced_ratio
from .encoders import ContentVecEncoder, RmvpeF0
from .extract import extract_phrase
from .loudness import loudness_manifest
from .shard import build_shard

_SAFE = re.compile(r"[^0-9A-Za-z_-]+")


def song_name(path: Path, wav_root: Path) -> str:
    """WAV のパスから曲名を作る。`{song}_{NNNN}` の `song` 部分になる。

    末尾が `_NNNN` だと `_song_of()` が曲名の一部を削ってしまうので、その形は避けます。
    """
    rel = path.relative_to(wav_root)
    stem = rel.parent.name if rel.parent != Path(".") else rel.stem
    if stem in ("", ".", "wav"):
        stem = rel.stem
    name = _SAFE.sub("_", stem).strip("_") or "song"
    if re.search(r"_\d{4,}$", name):
        name += "x"
    return name


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_extract(args, mel: MelSpec) -> Path:
    """1 段目。WAV -> `_cache/<phrase>.npz`。重いのでここだけ GPU を使う。"""
    import soundfile as sf

    wav_root = Path(args.wav_dir)
    wavs = sorted(p for p in wav_root.rglob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        sys.exit(f"WAV が見つかりません: {wav_root}")

    cache = Path(args.cache or Path(args.out) / "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    encoder = ContentVecEncoder(args.content_model, layer=args.layer, device=args.device)
    f0x = RmvpeF0(f0_min=args.f0_min, f0_max=args.f0_max, device=args.device)

    sources: dict[str, str] = {}
    n_phrases, n_skipped, t0 = 0, 0, time.time()
    for path in wavs:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        song = song_name(path, wav_root)
        spans = chunk_spans(len(wav), sr, chunk_sec=args.chunk_sec, min_sec=args.min_sec)
        kept = 0
        for a, b in spans:
            out = extract_phrase(np.ascontiguousarray(wav[a:b]), sr,
                                 content_encoder=encoder, f0_extract=f0x, mel=mel)
            # 無声だけの chunk を捨てる。曲を先頭から固定長で切るとイントロや間奏が
            # 丸ごと無声の phrase になり、学習に入れると「無音を出す」ことを学ぶ。
            if voiced_ratio(out["uv"]) < args.min_voiced:
                n_skipped += 1
                continue
            np.savez(cache / f"{song}_{kept:04d}.npz", **out)
            kept += 1
            n_phrases += 1
        sources[str(path.relative_to(wav_root))] = sha256_of(path)
        print(f"  {path.name[:44]:<44} {kept:>3} phrases "
              f"({len(spans) - kept} 無声で除外)", flush=True)

    (cache / "_sources.json").write_text(
        json.dumps({"wav_dir": str(wav_root), "sha256": sources,
                    "chunk_sec": args.chunk_sec, "min_sec": args.min_sec,
                    "min_voiced": args.min_voiced, "skipped_unvoiced": n_skipped,
                    "mel": mel.to_dict(),
                    **loudness_manifest(hop=mel.hop, n_fft=mel.n_fft),
                    **encoder.manifest(), **f0x.manifest()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[extract] {len(wavs)} files -> {n_phrases} phrases "
          f"({n_skipped} 無声で除外)  {time.time() - t0:.0f}s  -> {cache}")
    return cache


def stage_shard(args, mel: MelSpec, cache: Path) -> dict:
    """2 段目。`_cache/` -> `svc_shard.npz`。CPU だけで完結し、何度でも回せる。"""
    files = sorted(p for p in cache.glob("*.npz"))
    if not files:
        sys.exit(f"cache が空です: {cache}")
    phrases = {}
    for p in files:
        z = np.load(p)
        phrases[p.stem] = {k: z[k] for k in ("content", "f0_hz", "uv", "loudness", "mel")}

    extra = {}
    src = cache / "_sources.json"
    if src.exists():
        extra = json.loads(src.read_text(encoding="utf-8"))
    manifest = build_shard(phrases, args.out, n_dims=args.n_dims,
                           subset_seed=args.subset_seed, frame_rate=mel.frame_rate,
                           manifest_extra=extra)
    print(f"[shard] {manifest['n_phrases']} phrases / {manifest['total_frames']} frames "
          f"/ content_dim {manifest['n_dims']}  -> {args.out}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav-dir", help="入力 WAV ディレクトリ（再帰）")
    ap.add_argument("--out", required=True, help="出力先（data/<db>）")
    ap.add_argument("--cache", default=None, help="中間 cache（既定 <out>/_cache）")
    ap.add_argument("--from-cache", default=None,
                    help="1 段目を飛ばし、この cache から shard だけ作り直す")
    ap.add_argument("--n-dims", type=int, default=256, help="shard に書く content の次元")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--chunk-sec", type=float, default=10.0)
    ap.add_argument("--min-sec", type=float, default=2.0)
    ap.add_argument("--min-voiced", type=float, default=0.3,
                    help="有声フレームがこの割合未満の chunk は捨てる（無音のイントロ対策）")
    ap.add_argument("--content-model", default="lengyue233/content-vec-best")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--f0-min", type=float, default=65.0)
    ap.add_argument("--f0-max", type=float, default=1100.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N ファイルだけ（smoke 用）")
    args = ap.parse_args()

    mel = MelSpec()
    if args.from_cache:
        stage_shard(args, mel, Path(args.from_cache))
        return
    if not args.wav_dir:
        sys.exit("--wav-dir か --from-cache のどちらかが要ります")
    cache = stage_extract(args, mel)
    stage_shard(args, mel, cache)


if __name__ == "__main__":
    main()
