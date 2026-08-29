"""Download and unpack the Natsume (なつめ) singing database from Google Drive.

A single zip laid out by file type. This script downloads it, extracts with correct file
names, and places the folders under a clean path the preprocess recipe expects:

    download/natsume/wav/{song}.wav
    download/natsume/mono_label/{song}.lab      (lab = decimal seconds)
    download/natsume/xml/{song}.xml             (MusicXML)

    python preprocess/download_scripts/download_natsume.py               # download from Drive
    python preprocess/download_scripts/download_natsume.py --zip download/Natsume_Singing_DB_0713.zip   # local zip
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _gdrive import extract_cp932, fetch_zips, move_replace, sole_subdir

NATSUME_ID = "1M2BJwKUjGkiJ8_nn1CugG-IXoSSFbYUm"    # single zip on Google Drive
_NEEDED = ("wav", "mono_label", "xml")


def download_natsume(out_root: str = "download", work: str = "download/_tmp",
                     zip_path: str | None = None, keep_tmp: bool = False) -> Path:
    work_dir = Path(work) / "natsume"
    zips = fetch_zips(NATSUME_ID, work_dir, name="natsume", zip_path=zip_path)

    ex = work_dir / "extract"
    for z in zips:
        extract_cp932(z, ex)

    src = sole_subdir(ex, must_have=_NEEDED)          # Natsume_Singing_DB_0713/{wav,mono_label,xml}
    dst = Path(out_root) / "natsume"
    move_replace(src, dst)

    n_wav = len(list((dst / "wav").glob("*.wav")))
    n_lab = len(list((dst / "mono_label").glob("*.lab")))
    n_xml = len(list((dst / "xml").glob("*.xml")))
    print(f"natsume: -> {dst}  ({n_wav} wav, {n_lab} lab, {n_xml} xml)")
    if not keep_tmp and zip_path is None:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        shutil.rmtree(ex, ignore_errors=True)         # drop the extract copy, keep the zip
    return dst


def main():
    ap = argparse.ArgumentParser(description="Download the Natsume singing DB.")
    ap.add_argument("--out_root", default="download")
    ap.add_argument("--zip", dest="zip_path", default=None,
                    help="use an already-downloaded zip instead of fetching from Drive")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()
    download_natsume(out_root=args.out_root, zip_path=args.zip_path, keep_tmp=args.keep_tmp)


if __name__ == "__main__":
    main()
