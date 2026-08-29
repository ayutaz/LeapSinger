"""Shared helpers for the dataset downloaders: fetch a zip (or a folder of zips) from Google
Drive and extract it with correct shift-jis (cp932) file names, skipping macOS archive junk.

Japanese singing DBs are zipped on Windows/macOS with shift-jis entry names and no UTF-8 flag,
so generic unzip tools mojibake the folder names. `extract_cp932` decodes those names the way
the archive intended, and drops the `__MACOSX/` / `.DS_Store` clutter that macOS adds.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def fix_cp932(name: str) -> str:
    """Recover a shift-jis name that zipfile decoded as cp437 (standard Japanese-zip fix)."""
    try:
        return name.encode("cp437").decode("cp932")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def _is_junk(name: str) -> bool:
    parts = name.split("/")
    return parts[0] == "__MACOSX" or parts[-1] == ".DS_Store"


def extract_cp932(zip_path: Path, out_dir: Path) -> None:
    """Extract `zip_path` into `out_dir`, decoding non-UTF8 entry names as cp932 and skipping
    macOS junk (`__MACOSX/`, `.DS_Store`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            is_utf8 = bool(info.flag_bits & 0x800)
            name = info.filename if is_utf8 else fix_cp932(info.filename)
            if _is_junk(name):
                continue
            target = out_dir / name
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def fetch_zips(drive_id: str, work_dir: Path, *, is_folder: bool = False,
               name: str = "db", zip_path: str | None = None) -> list[Path]:
    """Return the zip(s) for a Drive source, downloading with gdown unless `zip_path` is given.

    `zip_path` lets you point at an already-downloaded zip (skips the network entirely).
    `is_folder=True` treats `drive_id` as a Drive folder and collects every .zip inside it.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if zip_path is not None:
        p = Path(zip_path)
        if not p.exists():
            raise SystemExit(f"--zip {p} does not exist")
        return [p]

    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown is required: pip install gdown")

    if is_folder:
        gdown.download_folder(id=drive_id, output=str(work_dir), quiet=False, use_cookies=False)
        zips = sorted(work_dir.rglob("*.zip"))
    else:
        z = work_dir / f"{name}.zip"
        got = gdown.download(id=drive_id, output=str(z), quiet=False)
        if got is None or not z.exists() or z.stat().st_size == 0:
            raise SystemExit(
                f"download failed for {name} (Drive id {drive_id}) — no zip written. "
                f"Check the link is public and not over its Google-Drive download quota.")
        zips = [z]
    if not zips:
        raise SystemExit(f"no zip downloaded for {name} (Drive id {drive_id})")
    return zips


def move_replace(src: Path, dst: Path) -> None:
    """Move `src` onto `dst`, replacing any existing `dst`."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def sole_subdir(root: Path, *, must_have=None) -> Path:
    """Return the single real (non-junk) sub-directory of `root`. If `must_have` is given, prefer
    a sub-directory that contains all of those relative paths (files or dirs)."""
    root = Path(root)
    subs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name != "__MACOSX"]
    if must_have:
        for p in subs:
            if all((p / rel).exists() for rel in must_have):
                return p
    if len(subs) == 1:
        return subs[0]
    raise FileNotFoundError(
        f"expected one dataset directory under {root}, found: {[p.name for p in subs]}")
