"""Download and unpack the ONIKU_KURUMI (おにくくるみ) singing database from Google Drive.

A single zip with per-song folders. This script downloads it, extracts with correct file
names, and places the songs under a clean path the preprocess recipe expects:

    download/oniku/{song}/{song}.{wav,lab,musicxml, ...}   (lab = HTS 100ns)

    uv run python preprocess/download_scripts/download_oniku.py                 # download from Drive
    uv run python preprocess/download_scripts/download_oniku.py --zip download/ONIKU_KURUMI_UTAGOE_DB.zip   # use a local zip
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _gdrive import extract_cp932, fetch_zips, move_replace, sole_subdir

ONIKU_ID = "17VOlqPKT7ssnOTCp6_ZBIWht1-h8FAt7"      # single zip on Google Drive


def _per_song_root(ex: Path) -> Path:
    """Directory whose children are song folders each holding {song}/{song}.wav."""
    top = sole_subdir(ex)                            # ONIKU_KURUMI_UTAGOE_DB/
    songs = [c for c in top.iterdir() if c.is_dir() and (c / f"{c.name}.wav").exists()]
    if not songs:
        raise FileNotFoundError(f"no per-song {{song}}/{{song}}.wav folders under {top}")
    return top


def download_oniku(out_root: str = "download", work: str = "download/_tmp",
                   zip_path: str | None = None, keep_tmp: bool = False) -> Path:
    work_dir = Path(work) / "oniku"
    zips = fetch_zips(ONIKU_ID, work_dir, name="oniku", zip_path=zip_path)

    ex = work_dir / "extract"
    for z in zips:
        extract_cp932(z, ex)

    src = _per_song_root(ex)
    dst = Path(out_root) / "oniku"
    move_replace(src, dst)

    songs = [c for c in dst.iterdir() if c.is_dir() and (c / f"{c.name}.wav").exists()]
    n_xml = sum((c / f"{c.name}.musicxml").exists() for c in songs)
    n_lab = sum((c / f"{c.name}.lab").exists() for c in songs)
    print(f"oniku: -> {dst}  ({len(songs)} songs, {n_xml} musicxml, {n_lab} lab)")
    if not keep_tmp and zip_path is None:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        import shutil
        shutil.rmtree(ex, ignore_errors=True)         # drop the extract copy, keep the zip
    return dst


def main():
    ap = argparse.ArgumentParser(description="Download the ONIKU_KURUMI singing DB.")
    ap.add_argument("--out_root", default="download")
    ap.add_argument("--zip", dest="zip_path", default=None,
                    help="use an already-downloaded zip instead of fetching from Drive")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()
    download_oniku(out_root=args.out_root, zip_path=args.zip_path, keep_tmp=args.keep_tmp)


if __name__ == "__main__":
    main()
