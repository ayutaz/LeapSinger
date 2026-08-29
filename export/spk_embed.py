"""Speaker-embedding export for multi-speaker models (HarmonicAcousticModelMultiSpk).

The multispk model adds a per-speaker vector `spk_proj(spk_bank.weight[i])` (dim = hidden) to the
condition at every frame (`_spk_add`). For ONNX we expose that H-dim vector as the DiffSinger
`spk_embed` graph input (host mixes speakers, the graph adds the mixed vector). Each speaker's
vector is also written as a raw little-endian float32 `.emb` (DiffSinger attachment format), so a
host can load/mix them exactly as `_spk_add` would.

A single-speaker model (HarmonicAcousticModel, no spk_bank) has no speaker embedding — nothing
to export; WrapperA uses speaker='none'.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def has_speakers(model) -> bool:
    return getattr(model, "spk_bank", None) is not None


def speaker_vector(model, spk_id: int) -> np.ndarray:
    """The H-dim vector `_spk_add` would add for `spk_id` (== spk_proj(spk_bank[spk_id]))."""
    assert has_speakers(model), "model has no speaker bank"
    with torch.no_grad():
        idx = torch.tensor([int(spk_id)], device=model.spk_bank.weight.device)
        v = model.spk_proj(model.spk_bank(idx))[0]         # [hidden]
    return v.detach().cpu().numpy().astype(np.float32)


def write_emb(path: str, vec: np.ndarray) -> str:
    """Write an H-dim vector as a raw little-endian float32 blob (DiffSinger `.emb`)."""
    np.asarray(vec, dtype="<f4").tofile(path)
    return path


def export_speaker_embeds(model, out_dir: str, model_name: str, spk_names: dict) -> list[str]:
    """Write `{model_name}.{name}.emb` for each speaker. `spk_names` = {spk_id: name}.
    Returns the list of `{model_name}.{name}` speaker keys (for dsconfig 'speakers')."""
    keys = []
    for sid, name in sorted(spk_names.items()):
        write_emb(os.path.join(out_dir, f"{model_name}.{name}.emb"), speaker_vector(model, sid))
        keys.append(f"{model_name}.{name}")
    return keys
