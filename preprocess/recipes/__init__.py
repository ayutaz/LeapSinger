"""Per-singer preprocessing hooks.

`configs/recipes/{db}.yaml` holds each database's *declarative* config (paths, speaker id,
F0 range). This package holds each database's *code* — optional, singer-specific steps that
should not live in the shared pipeline. A module here is loaded by name (the recipe's
`hook`, defaulting to its `name`); if it defines `fix_lab(song, intervals)`, the pipeline runs
it over that song's raw lab intervals before building the phoneme timeline. A database with
nothing to fix simply has no module here.
"""
from __future__ import annotations

import glob as _glob
import importlib
import os as _os


def load(name: str):
    """Return the recipe hook module `recipes.{name}`, or None if there is no such module."""
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError:
        return None


def resolve_song_file(db_root: str, template: str, song: str) -> str:
    """Resolve one of a song's files from a `{song}/...{song}.ext` template.

    Returns the exact templated path when it exists. Some songs ship with a folder name that
    differs from the file stem (e.g. `futaridakeno_silence/futaridakenosilence.musicxml`,
    `future_sky6/FutureSky6.lab`); there the exact path is missing, so fall back to the sole
    file of that extension inside the song folder. Raw data is never renamed."""
    p = _os.path.join(db_root, template.format(song=song))
    if _os.path.exists(p):
        return p
    ext = _os.path.splitext(template)[1]
    folder = _os.path.dirname(_os.path.join(db_root, template.format(song=song)))
    cands = sorted(_glob.glob(_os.path.join(folder, "*" + ext)))
    return cands[0] if cands else p


def apply_lab_fixes(intervals: list, fixes: list, *, eps: float = 0.005) -> list:
    """Apply lab corrections to (start_sec, end_sec, phoneme) intervals — helper for hook modules.

    Each fix targets ONE interval by its start time (`at`, seconds) and, as a reproducibility
    guard, its current phoneme (`from`):

        {at: 74.822,  from: cl, set: a}         relabel the phoneme
        {at: 102.021, from: w,  merge: next}    delete it, extend the NEXT interval back over its span
        {at: 196.950, from: e,  merge: prev}    delete it, extend the PREV interval forward over its span
        {at: 81.222,  from: a,  split: 81.74}   split it at 81.74s into two (both keep the phoneme;
                                                add `set` to relabel the halves)
        {at: 44.96, insert: w, end: 45.06}      insert a new phoneme over [44.96, 45.06], carving
                                                the span out of the neighbouring intervals

    `from` is asserted so a fix raises loudly if the downloaded lab ever changes, rather than
    silently editing the wrong phoneme.
    """
    iv = [list(x) for x in intervals]
    for fx in fixes:
        if "insert" in fx:                                # insert a new phoneme spanning [at, end],
            t0, t1, ph = float(fx["at"]), float(fx["end"]), fx["insert"]   # carving it out of neighbours
            if not t0 < t1:
                raise ValueError(f"lab_fix {fx}: need at < end")
            out = []
            for s, e, p in iv:
                if e <= t0 or s >= t1:
                    out.append([s, e, p])                 # fully outside the inserted span
                    continue
                if s < t0:
                    out.append([s, t0, p])                # keep the part before the insert
                if e > t1:
                    out.append([t1, e, p])                # keep the part after the insert
            out.append([t0, t1, ph])
            iv = sorted(out, key=lambda x: x[0])
            continue
        at, want = float(fx["at"]), fx.get("from")
        hits = [i for i, (s, _e, p) in enumerate(iv)
                if abs(s - at) <= eps and (want is None or p == want)]
        if len(hits) != 1:
            raise ValueError(f"lab_fix {fx}: matched {len(hits)} intervals near {at}s "
                             f"(from={want!r}); expected exactly 1")
        i = hits[0]
        if "set" in fx:
            iv[i][2] = fx["set"]
        elif fx.get("merge") == "next":
            if i + 1 >= len(iv):
                raise ValueError(f"lab_fix {fx}: no next interval to merge into")
            iv[i + 1][0] = iv[i][0]
            del iv[i]
        elif fx.get("merge") == "prev":
            if i - 1 < 0:
                raise ValueError(f"lab_fix {fx}: no prev interval to merge into")
            iv[i - 1][1] = iv[i][1]
            del iv[i]
        elif "split" in fx:
            t = float(fx["split"])
            s, e, p = iv[i]
            if not (s < t < e):
                raise ValueError(f"lab_fix {fx}: split time {t}s not inside [{s}, {e}]")
            new = fx.get("set", p)                        # optional relabel of both halves
            iv[i] = [s, t, new]
            iv.insert(i + 1, [t, e, new])
        else:
            raise ValueError(f"lab_fix {fx}: needs 'set', 'merge: prev|next', or 'split: <sec>'")
    return [tuple(x) for x in iv]
