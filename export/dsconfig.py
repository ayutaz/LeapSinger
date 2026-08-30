"""dsconfig.yaml + phonemes.txt writers (DiffSinger/OpenUTAU attachment format).

The mel spec here is what OpenUTAU needs to interpret the acoustic model's mel and pair it with
the (separately exported NHVSing) vocoder: e-based ln-mel, slaney scale, 128 bins, 40–16000 Hz,
fft/win 2048, hop 512 (pseudo-512 export) or 256. All DiffSinger variance embeds are declared
false (LeapSinger supports none of breathiness/tension/energy/voicing, nor gender/velocity).

LeapSinger divergence from stock DiffSinger (recorded in a sidecar, NOT in dsconfig.yaml so a
DiffSinger parser never trips on unknown keys): our acoustic ONNX is a single end-to-end graph
(tokens/durations/f0 -> mel) with the rectified-flow Euler loop baked in — there is no separate
diffusion.onnx and no external speedup/steps input. OpenUTAU's stock DiffSinger renderer drives
a 2-file (fs2 + diffusion) split; wiring this single-shot graph into OpenUTAU is the real-device
milestone and may need either a thin adapter or the 2-file split.
"""
from __future__ import annotations

import json
import os

import yaml

from leapsinger.mel import F_MAX, F_MIN, N_FFT, N_MELS, SR, WIN_LEN

from . import openutau_assets as oua


def write_phonemes(out_dir: str, model_name: str, phonemes=None) -> str:
    """Write `{model_name}.phonemes.txt` — one phoneme per line, line index == token id, with
    pau->SP / br->AP (OpenUTAU's LoadPhonemes reads a .txt list or a {name:index} .json object,
    NOT a JSON list). `phonemes` = the model's list (from the ckpt); default = bundled Japanese.
    See openutau_assets.write_phonemes_txt."""
    return oua.write_phonemes_txt(out_dir, filename=f"{model_name}.phonemes.txt", phonemes=phonemes)


def write_dsconfig(out_dir: str, model_name: str, *, hop: int, hidden_size: int,
                   vocoder_name: str, speakers=None, acoustic_file: str | None = None) -> str:
    """Write dsconfig.yaml. `speakers` = list of `{model_name}.{spk}` names (embed mode, multi-spk)
    or None. `acoustic_file` overrides the onnx filename (e.g. `{model}.{spk}.onnx` for baked spk)."""
    assert hop in (256, 512), hop
    cfg = {
        "phonemes": f"{model_name}.phonemes.txt",
        "acoustic": acoustic_file or f"{model_name}.onnx",
        "hidden_size": int(hidden_size),
        "vocoder": vocoder_name,
        # mel specification (matches leapsinger/mel.py + the vocoder)
        "sample_rate": int(SR),
        "hop_size": int(hop),
        "win_size": int(WIN_LEN),
        "fft_size": int(N_FFT),
        "num_mel_bins": int(N_MELS),
        "mel_fmin": float(F_MIN),
        "mel_fmax": float(F_MAX),
        "mel_base": "e",          # ln-mel (our mel is natural-log); DiffSinger 'e'
        "mel_scale": "slaney",    # librosa.filters.mel default (== our _mel_basis)
        # capabilities (LeapSinger supports none of these)
        "use_lang_id": False,
        "use_key_shift_embed": False,
        "use_speed_embed": False,
        "use_breathiness_embed": False,
        "use_tension_embed": False,
        "use_energy_embed": False,
        "use_voicing_embed": False,
    }
    if speakers:
        cfg["speakers"] = list(speakers)
    path = os.path.join(out_dir, "dsconfig.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def write_leapsinger_meta(out_dir: str, model_name: str, *, variant: str, hop: int,
                          use_uv: bool, infer_steps: int, speaker_mode: str) -> str:
    """Sidecar with LeapSinger-specific facts a DiffSinger parser doesn't understand."""
    meta = {
        "model_name": model_name,
        "variant": variant,                 # 'diffsinger' (A) | 'full' (B)
        "hop": hop,
        "end_to_end": True,                 # flow Euler loop baked in; no external diffusion/speedup
        "infer_steps": int(infer_steps),
        "use_uv": bool(use_uv),
        "speaker_mode": speaker_mode,       # none | embed | bake
    }
    path = os.path.join(out_dir, f"{model_name}.leapsinger.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path
