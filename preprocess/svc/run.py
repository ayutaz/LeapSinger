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
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from leapsinger.config import MelSpec

from .chunk import chunk_spans, voiced_ratio
from .encoders import ContentVecEncoder, RmvpeF0
from .extract import extract_phrase
from .loudness import loudness_manifest
from .shard import build_shard

# npz の key（`<name>|content`）とファイル名に使えない文字だけを落とす。
# **CJK は残す。** ASCII だけに削ると日本語題の曲がすべて同じ名前に潰れ、曲単位 split が
# 効かなくなる（実測: GTSinger の 1 歌手 1,922 ファイル中 1,723 件が 1 つの名前に潰れた）。
_SAFE = re.compile(r"[^\w-]+", re.UNICODE)


def song_name(path: Path, wav_root: Path, parts: Sequence[int] | None = None) -> str:
    """WAV のパスから曲名を作る。`{song}_{NNNN}` の `song` 部分になる。

    既定は親ディレクトリ名です（`<曲>/<曲>.wav` という配置を想定）。**入れ子の深い
    コーパスでは `parts` で曲にあたる階層を指定します。** 例えば GTSinger の 1 歌手ぶんを
    root にすると相対パスが `<技法>/<曲>/<Group>/0000.wav` になるので `parts=(1,)` です。
    ここを親ディレクトリのままにすると、別の曲がすべて `Control_Group` に潰れます
    （実測: 40 ファイルが 3 名に潰れた）。

    技法や Group をまたいで**同じ曲は同じ song 名**にすること。別々にすると、同じ曲が
    train と eval に分かれて leakage します。

    末尾が `_NNNN` だと `_song_of()` が曲名の一部を削ってしまうので、その形は避けます。
    """
    rel = path.relative_to(wav_root)
    if parts:
        dirs = rel.parts[:-1]
        picked = [dirs[i] for i in parts if -len(dirs) <= i < len(dirs)]
        stem = "_".join(picked) if picked else rel.stem
    else:
        stem = rel.parent.name if rel.parent != Path(".") else rel.stem
    if stem in ("", ".", "wav"):
        stem = rel.stem
    # 表記ゆれを畳む。GTSinger の JA-Tenor-1 は `Heartful_Song` と `Heartful_song` が
    # 同居しており、別の曲として扱うと同じ曲が train と eval に分かれて leakage する。
    name = _SAFE.sub("_", stem.casefold()).strip("_") or "song"
    if re.search(r"_\d{4,}$", name):
        name += "x"
    return name


def phrase_name(counters: dict[str, int], song: str) -> str:
    """`{song}_{NNNN}` を採番する。**曲ごとに通し番号**で、ファイルをまたいで続ける。

    ファイルごとに 0 へ戻すと、同じ曲の別ファイルが同じ名前になり、cache を**黙って
    上書き**します。例外にもならず shard のフレーズ数が減るだけなので気づけません。
    """
    idx = counters.get(song, 0)
    counters[song] = idx + 1
    return f"{song}_{idx:04d}"


def select_by_budget(entries, max_seconds):
    """`[(song, path, seconds), ...]` から合計が `max_seconds` を超えない範囲で選ぶ。

    **曲をまたいでラウンドロビン**で取ります。先頭から順に取ると技法や曲が
    アルファベット順に偏るためです。`max_seconds` が偽なら全件返します。

    base 事前学習で効くのは総時間より**話者の多様性**です（[実行計画](../../doc/svc-plan.md) M3）。
    1 歌手あたりの上限を決めることで、同じ抽出コストで話者数を増やせます。
    """
    if not max_seconds or max_seconds <= 0:
        return list(entries)
    by_song: dict[str, list] = {}
    for e in entries:
        by_song.setdefault(e[0], []).append(e)
    picked, total = set(), 0.0
    order = sorted(by_song)
    round_i = 0
    while True:
        added = False
        for song in order:
            files = by_song[song]
            if round_i >= len(files):
                continue
            e = files[round_i]
            if total + e[2] > max_seconds:
                continue
            picked.add(id(e))
            total += e[2]
            added = True
        if not added:
            break
        round_i += 1
    return [e for e in entries if id(e) in picked]


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
    if args.max_hours:
        # 歌手ごとの分量を揃える。曲をまたいでラウンドロビンで選ぶ（先頭から取ると偏る）。
        entries = [(song_name(p, wav_root, args.song_parts), p, sf.info(p).duration)
                   for p in wavs]
        total_h = sum(e[2] for e in entries) / 3600
        wavs = [e[1] for e in select_by_budget(entries, args.max_hours * 3600)]
        print(f"[budget] {total_h:.2f}h -> {args.max_hours:.2f}h 上限で "
              f"{len(entries)} -> {len(wavs)} files", flush=True)

    cache = Path(args.cache or Path(args.out) / "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    encoder = ContentVecEncoder(args.content_model, layer=args.layer, device=args.device)
    f0x = RmvpeF0(f0_min=args.f0_min, f0_max=args.f0_max, device=args.device)

    sources: dict[str, str] = {}
    counters: dict[str, int] = {}          # 曲ごとの通し番号。ファイルをまたいで続ける
    written: set[str] = set()
    n_phrases, n_skipped, t0 = 0, 0, time.time()
    for path in wavs:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        song = song_name(path, wav_root, args.song_parts)
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
            name = phrase_name(counters, song)
            if name in written:            # 起きないはずだが、起きたら黙って上書きさせない
                sys.exit(f"phrase 名が衝突しました: {name}（--song-parts を見直してください）")
            written.add(name)
            np.savez(cache / f"{name}.npz", **out)
            kept += 1
            n_phrases += 1
        sources[str(path.relative_to(wav_root))] = sha256_of(path)
        if args.verbose or len(wavs) <= 200:
            print(f"  {path.name[:44]:<44} {kept:>3} phrases "
                  f"({len(spans) - kept} 無声で除外)  [{song}]", flush=True)
        elif len(sources) % 200 == 0:
            print(f"  ... {len(sources)}/{len(wavs)} files, {n_phrases} phrases, "
                  f"{time.time() - t0:.0f}s", flush=True)

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
    ap.add_argument("--song-parts", default=None,
                    help="曲名にする相対パス階層をカンマ区切りで（0 始まり、負も可）。"
                         "既定は親ディレクトリ名。GTSinger の 1 歌手ぶんを --wav-dir に"
                         "したときは `1`（<技法>/<曲>/<Group>/x.wav の <曲>）")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="この歌手から使う音声の上限（時間）。曲をまたいで均等に選ぶ")
    ap.add_argument("--verbose", action="store_true", help="ファイルごとの内訳を必ず出す")
    args = ap.parse_args()
    args.song_parts = (tuple(int(x) for x in args.song_parts.split(","))
                       if args.song_parts else None)

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
