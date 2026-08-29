"""OpenUTAU / DiffSinger phoneme-set assets shared by the acoustic, duration and pitch models.

Two files, both derived from LeapSinger's canonical phoneme set + Japanese kana table:

  * phonemes.txt  — one phoneme per line; the LINE NUMBER is the token index. This is the format
    OpenUTAU's `DiffSingerUtils.LoadPhonemes` reads (a `.json` would have to be a {name:index}
    OBJECT (a {name: index} map), not a list — a plain list form would be unparseable).
    The only renaming vs LeapSinger's internal vocab: index 0 `pau` -> `SP` and `br` -> `AP`,
    because OpenUTAU hardcodes SP (silence) / AP (breath) and looks them up by string in the
    phoneme set. The model only ever sees the INDEX, so this is a pure label alias — no retrain.

  * dsdict.yaml   — the G2p dictionary OpenUTAU's DiffSinger phonemizer consumes:
        symbols: [{symbol: a, type: vowel}, {symbol: k, type: stop}, ...]
        entries: [{grapheme: あ, phonemes: [a]}, {grapheme: か, phonemes: [k, a]}, ...]
    `type: vowel` marks vowels (is_vowel); `semivowel`/`liquid` mark glides; anything else is a
    plain consonant. Built from dict/kana2phonemes.table.

NOTE: a full OpenUTAU Japanese DiffSinger voicebank ALSO needs a Japanese DiffSinger phonemizer
(C#, not present in OpenUTAU) plus the duration + pitch ONNX models (separate LeapSinger repos).
These two files are the shared phoneme-set foundation all of those must agree on.
"""
from __future__ import annotations

import os

from preprocess.vocab import CANONICAL_PHONEMES

# LeapSinger internal name -> OpenUTAU name (index preserved; pure alias)
_OU_RENAME = {"pau": "SP", "br": "AP"}

# phoneme -> dsdict `type`. vowel => is_vowel; semivowel/liquid => glide; else consonant.
_VOWELS = {"a", "i", "u", "e", "o", "A", "I", "U", "O", "N", "EP", "E"}
_SEMIVOWELS = {"y", "w"}
_LIQUIDS = {"r", "ry"}

# kana-table phonemes the model's canonical set lacks -> nearest canonical fallback.
# vy (palatalized v, only ヴュ/ヴゅ — rare loanword) has no canonical form; drop the palatal to v.
_PHONEME_FALLBACK = {"vy": "v"}


def openutau_phoneme(name: str) -> str:
    return _OU_RENAME.get(name, name)


def symbol_type(name: str) -> str:
    if name in _VOWELS:
        return "vowel"
    if name in _SEMIVOWELS:
        return "semivowel"
    if name in _LIQUIDS:
        return "liquid"
    return "consonant"


def write_phonemes_txt(out_dir: str, filename: str = "phonemes.txt", phonemes=None) -> str:
    """phonemes.txt: one phoneme per line, line index == token index (pau->SP, br->AP).
    `phonemes` = the model's phoneme list (from the checkpoint); default = the bundled Japanese set."""
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for name in (phonemes or CANONICAL_PHONEMES):
            f.write(openutau_phoneme(name) + "\n")
    return path


def _parse_kana_table(table_path: str):
    """kana2phonemes.table lines: `<grapheme> <ph1> <ph2> ...`. Returns [(grapheme, [phonemes])].
    Silence/breath rows are folded to OpenUTAU's SP/AP; pure-silence rows (pau/sil) are dropped
    (OpenUTAU inserts SP itself). Duplicate graphemes keep the first mapping."""
    from preprocess.vocab import CANONICAL_PHONEMES
    canon = set(CANONICAL_PHONEMES) | {"SP", "AP"}
    entries, seen, dropped = [], set(), []
    with open(table_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            g, phs = parts[0], parts[1:]
            phs = [_OU_RENAME.get(p, p) for p in phs]           # br->AP, (pau->SP)
            phs = [_PHONEME_FALLBACK.get(p, p) for p in phs]    # vy->v, ...
            if all(p == "SP" for p in phs) or phs == ["sil"]:   # pure silence -> OpenUTAU's job
                continue
            unknown = [p for p in phs if p not in canon]
            if unknown:                                          # references a phoneme the model lacks
                dropped.append((g, phs)); continue
            if g in seen:
                continue
            seen.add(g)
            entries.append((g, phs))
    if dropped:
        print(f"[openutau_assets] dropped {len(dropped)} dsdict entries with non-canonical phonemes: "
              + ", ".join(f"{g}->{'/'.join(p)}" for g, p in dropped[:8])
              + (" ..." if len(dropped) > 8 else ""))
    return entries


def default_kana_table() -> str:
    """Path to the bundled kana->phoneme table (dict/)."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "dict", "kana2phonemes.table")


def write_dsdict(out_dir: str, table_path: str | None = None, filename: str = "dsdict.yaml") -> str:
    """Convenience: write dsdict.yaml from the bundled (or given) kana table."""
    return write_dsdict_yaml(out_dir, table_path or default_kana_table(), filename=filename)


def write_dsdict_yaml(out_dir: str, table_path: str, filename: str = "dsdict.yaml") -> str:
    """dsdict.yaml (symbols + entries) from the kana table. Symbols = every phoneme that appears
    in an entry, classified by type (SP/AP excluded — OpenUTAU adds them after loading the dict)."""
    import yaml
    entries = _parse_kana_table(table_path)
    used = {p for _, phs in entries for p in phs if p not in ("SP", "AP")}
    # keep a stable, readable symbol order = canonical order, restricted to used symbols
    symbols = [s for s in map(openutau_phoneme, CANONICAL_PHONEMES) if s in used]  # used already excludes SP/AP
    data = {
        "symbols": [{"symbol": s, "type": symbol_type(s)} for s in symbols],
        # some graphemes start with ' (Edge) or ・ (GlottalStop) -> use safe_dump so they are quoted
        "entries": [{"grapheme": g, "phonemes": list(phs)} for g, phs in entries],
    }
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return path
