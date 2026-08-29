"""Natsume-specific preprocessing.

The human-labelled Natsume lab carries a few phoneme slips (each confirmed against the audio
in Praat). We correct them here, in this singer's own module, so the shared aligner sees the
right phonemes — this stays out of the common pipeline because it is this database alone.
"""
from __future__ import annotations

from . import apply_lab_fixes

# song -> lab corrections. Times are the lab entry's start in seconds; `from` asserts the
# current phoneme so the fix fails loudly if the download ever changes.
LAB_FIXES = {
    "1": [
        {"at": 74.8217, "from": "cl", "set": "a"},       # 「まあす」の「あ」が促音 cl と誤ラベル → a
        {"at": 102.021, "from": "w",  "merge": "next"},  # 「くうお」の「う」に無い w が付与 → 削除して u へ併合
        {"at": 196.950, "from": "e",  "merge": "prev"},  # 「みすえて」の「え」が e 二重 → 1つの e に統合
    ],
    "3": [
        {"at": 34.45,   "from": "s",  "set": "z"},        # 「ぞ」が s o と誤ラベル → z o
    ],
    "8": [
        {"at": 21.9094, "from": "w",  "merge": "next"},   # 助詞「を」を o と歌唱、labの w を削除 → ne o su
    ],
    "50": [
        {"at": 75.8979, "from": "o",  "merge": "next"},   # 「わあんさ」の o を次の a へ統合 → w a N s a
    ],
    "10": [
        {"at": 81.2222, "from": "a",  "split": 81.49},    # 「ああ」を2音符で歌唱、1つの a を2つに分割
    ],
}


def fix_lab(song: str, intervals: list) -> list:
    """Correct one song's raw lab intervals (no-op for songs without fixes)."""
    fixes = LAB_FIXES.get(song)
    return apply_lab_fixes(intervals, fixes) if fixes else intervals
