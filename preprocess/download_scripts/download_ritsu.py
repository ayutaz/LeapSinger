"""Download and unpack a 波音リツ (Namine Ritsu) singing database from Google Drive.

Three voice types are available, each a single zip:
    kire   (無印 / キレ)  — the default
    normal (ノーマル)
    soft   (ソフト)

The zips carry shift-jis (cp932) file names with no UTF-8 flag, so generic unzip tools
(macOS Archive Utility, etc.) mojibake the folder names. This script downloads with gdown and
re-extracts, decoding entry names as cp932 (see scripts/_gdrive.py), then places the song
folders under a clean ASCII path so the preprocess recipes can find them:

    kire   -> download/ritsu/DATABASE/{song}/{song}.{lab,musicxml,wav,...}
    normal -> download/ritsu_normal/DATABASE/{song}/...
    soft   -> download/ritsu_soft/DATABASE/{song}/...

    uv run python preprocess/download_scripts/download_ritsu.py --voice kire
    uv run python preprocess/download_scripts/download_ritsu.py --voice all
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _gdrive import extract_cp932, fetch_zips, move_replace

# voice type -> (google-drive id, is_folder). Each download is a single zip.
RITSU_SOURCES = {
    "kire":   ("1XA2cm3UyRpAk_BJb1LTytOWrhjsZKbSN", True),   # 無印 (default) — folder link
    "normal": ("1n-_ru4z7LV5JAQeyaMNlmjXBAFpjvR4A", False),
    "soft":   ("1AwGv3cv-NQX_yYnBxzmtYan36AxEy_0G", False),
}
_OUT_NAME = {"kire": "ritsu", "normal": "ritsu_normal", "soft": "ritsu_soft"}


def _find_database(root: Path) -> Path:
    """Locate the DATABASE / DATABASE_soft directory (holds per-song folders with a .wav)."""
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name.upper().startswith("DATABASE"):
            return p
    # fallback: a directory whose children are song folders each containing a .wav
    for p in sorted(root.rglob("*")):
        if p.is_dir() and any((c / f"{c.name}.wav").exists() for c in p.iterdir() if c.is_dir()):
            return p
    raise FileNotFoundError(f"no DATABASE dir found under {root}")


def download_ritsu(voice: str = "kire", out_root: str = "download",
                  work: str = "download/_tmp", keep_tmp: bool = False) -> Path:
    drive_id, is_folder = RITSU_SOURCES[voice]
    work_dir = Path(work) / voice
    zips = fetch_zips(drive_id, work_dir, is_folder=is_folder, name=voice)

    ex = work_dir / "extract"
    for z in zips:
        extract_cp932(z, ex)

    db = _find_database(ex)
    dst = Path(out_root) / _OUT_NAME[voice] / "DATABASE"
    move_replace(db, dst)

    n_songs = sum(1 for c in dst.iterdir() if c.is_dir())
    print(f"ritsu/{voice}: -> {dst}  ({n_songs} songs)")
    if not keep_tmp:
        shutil.rmtree(work_dir, ignore_errors=True)
    return dst


def main():
    ap = argparse.ArgumentParser(description="Download a Namine Ritsu singing DB.")
    ap.add_argument("--voice", default="kire",
                    choices=list(RITSU_SOURCES) + ["all"], help="voice type (default kire=無印)")
    ap.add_argument("--out_root", default="download")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()
    voices = list(RITSU_SOURCES) if args.voice == "all" else [args.voice]
    for v in voices:
        download_ritsu(v, out_root=args.out_root, keep_tmp=args.keep_tmp)


if __name__ == "__main__":
    main()
