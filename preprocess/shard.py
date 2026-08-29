"""Combine per-phrase npz files into one `shard.npz` per dataset.

Thousands of tiny npz files are slow to read (especially on networked drives), so we
pack them into a single `{name}|{field}` shard (the field length is also written to
`metadata.json` so the loader can bucket by length without reading the mel). The raw
per-phrase clip audio (present when preprocessing with `save_wav=True`) is split into a
`shard_wav.npz` sidecar so the main shard stays lean — the sidecar is only needed for
pitch augmentation.

metadata.json records the phoneme vocabulary and the frame/mel settings used, so the
dataset loader and model can be rebuilt to match the data exactly.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from leapsinger.config import MelSpec
from .vocab import DEFAULT_VOCAB, Vocab


def combine_db(db_dir: str, mel: MelSpec | None = None, vocab=None) -> int:
    """Pack `{db_dir}/phrases/*.npz` -> `{db_dir}/shard.npz` (+ `shard_wav.npz`) + metadata."""
    mel = mel or MelSpec()
    vocab = vocab or DEFAULT_VOCAB
    phrase_dir = os.path.join(db_dir, "phrases")
    files = sorted(glob.glob(os.path.join(phrase_dir, "*.npz")))
    if not files:
        print(f"  [skip] no phrases in {phrase_dir}")
        return 0

    shard: dict = {}
    wavs: dict = {}
    lengths: dict = {}
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        with np.load(p, allow_pickle=False) as d:
            for k in d.files:
                if k == "wav":
                    wavs[f"{name}|wav"] = d[k]            # -> shard_wav.npz sidecar
                else:
                    shard[f"{name}|{k}"] = d[k]
            lengths[name] = int(d["mel"].shape[1]) if "mel" in d.files else 0

    np.savez(os.path.join(db_dir, "shard.npz"), **shard)   # uncompressed = faster reads
    if wavs:
        np.savez(os.path.join(db_dir, "shard_wav.npz"), **wavs)

    meta_path = os.path.join(db_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta.update(
        phrases=lengths,
        n_phrases=len(lengths),
        n_phonemes=vocab.n_phonemes,
        phonemes=vocab.phonemes,               # ordered list (id = index); the vocab used here
        phoneme_vocab=vocab.phoneme2id,
        sample_rate=mel.sr,
        hop_size=mel.hop,
        frame_rate=mel.frame_rate,
        n_mels=mel.n_mels,
    )
    # speaker/style identity is not stored in the shard — it is assigned per-DB at train time.
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    sz = os.path.getsize(os.path.join(db_dir, "shard.npz")) / 1e6
    wtag = f" + shard_wav.npz" if wavs else ""
    print(f"  {os.path.basename(db_dir)}: {len(files)} phrases -> shard.npz ({sz:.0f}MB){wtag}")
    return len(files)


def main():
    ap = argparse.ArgumentParser(description="Combine per-phrase npz into a shard.")
    ap.add_argument("--db_dir", default=None, help="one dataset dir (contains phrases/)")
    ap.add_argument("--root", default="data", help="root for --all")
    ap.add_argument("--all", action="store_true", help="process every dataset under root")
    ap.add_argument("--phonemes", default=None,
                    help="phoneme-list file (default: bundled Japanese dict/ja.phonemes)")
    args = ap.parse_args()
    vocab = Vocab.load(args.phonemes)
    if args.all:
        total = 0
        for d in sorted(glob.glob(os.path.join(args.root, "*"))):
            if os.path.isdir(os.path.join(d, "phrases")):
                total += combine_db(d, vocab=vocab)
        print(f"done. total {total} phrases across datasets.")
    elif args.db_dir:
        combine_db(args.db_dir, vocab=vocab)
    else:
        ap.error("specify --db_dir or --all")


if __name__ == "__main__":
    main()
