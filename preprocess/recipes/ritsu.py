"""Ritsu-specific preprocessing.

Corrections confirmed against the audio (the singer's actual pronunciation):
  * LAB_FIXES — phoneme-label slips in the human .lab (see apply_lab_fixes for the ops).
"""
from __future__ import annotations

from . import apply_lab_fixes

# song -> lab corrections (times are the lab entry's start in seconds; `from` asserts the phoneme).
LAB_FIXES = {
    "1st_color": [
        {"at": 44.96, "insert": "w", "end": 45.06},   # 「うぉ」歌唱: う と お の間に w グライドを挿入
    ],
}


def fix_lab(song: str, intervals: list) -> list:
    fixes = LAB_FIXES.get(song)
    return apply_lab_fixes(intervals, fixes) if fixes else intervals
