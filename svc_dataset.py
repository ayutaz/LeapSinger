"""Dataset contract for precomputed offline SVC features.

Each data directory contains ``metadata.json`` and ``svc_shard.npz``. For every
phrase name in ``metadata.json['phrases']`` the shard stores:

``<name>|content``   float32 [T, content_dim], aligned to the mel frame grid
``<name>|f0_interp`` float32 [T] in Hz, gap-filled
``<name>|uv``        float32 [T], 1 for voiced
``<name>|loudness``  float32 [T], log-RMS or another consistent log-amplitude
``<name>|mel``       float32 [n_mels, T], NHVSing-compatible target mel

Heavy SSL and F0 models therefore run once during preprocessing, not in every
training step.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import _open, _song_of


class SVCFeatureDataset(Dataset):
    def __init__(
        self,
        dirs,
        split: str = "train",
        eval_songs: int = 2,
        min_sec: float = 0.0,
        seed: int = 42,
        spk_map: dict | None = None,
        style_map: dict | None = None,
    ):
        self.spk_map = spk_map
        self.style_map = style_map
        self.split = split
        self._cache: dict = {}
        self.files: list[tuple[str, str, str]] = []
        self.frame_counts: list[int] = []
        self.content_dim: int | None = None
        frame_rate = None

        for directory in dirs:
            directory = Path(directory)
            with open(directory / "metadata.json", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            shard = directory / "svc_shard.npz"
            if not shard.exists():
                raise FileNotFoundError(f"missing SVC feature shard: {shard}")
            frame_rate = float(metadata.get("frame_rate", frame_rate or 172.265625))
            phrases = metadata["phrases"]
            names = sorted(phrases)
            songs = sorted({_song_of(name) for name in names})
            n_hold = min(eval_songs, max(0, len(songs) - 1))
            hold = set(random.Random(seed).sample(songs, n_hold)) if n_hold else set()
            for name in names:
                is_eval = _song_of(name) in hold
                if (split == "eval") != is_eval:
                    continue
                frames = int(phrases[name])
                if min_sec and frames < min_sec * frame_rate:
                    continue
                self.files.append((str(shard), name, str(directory)))
                self.frame_counts.append(frames)

            declared_dim = metadata.get("content_dim")
            if declared_dim is not None:
                declared_dim = int(declared_dim)
                if self.content_dim not in (None, declared_dim):
                    raise ValueError("all SVC datasets must use the same content_dim")
                self.content_dim = declared_dim

        self.frame_rate = float(frame_rate or 172.265625)
        if self.files and self.content_dim is None:
            shard, name, _ = self.files[0]
            z = _open(shard, self._cache)
            self.content_dim = int(z[f"{name}|content"].shape[-1])
        self.content_dim = int(self.content_dim or 0)
        self.speaker_ids = [self._mapped_id(db, self.spk_map) for _, _, db in self.files]
        self.db_ids = [os.path.basename(os.path.normpath(db)) for _, _, db in self.files]
        print(
            f"[svc-dataset] {split}: {len(self.files)} phrases  "
            f"frame_rate={self.frame_rate:.3f}  content_dim={self.content_dim}"
        )

    @staticmethod
    def _mapped_id(dbdir: str, mapping: dict | None) -> int:
        if not mapping:
            return 0
        return int(mapping.get(os.path.basename(os.path.normpath(dbdir)), 0))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        shard, name, dbdir = self.files[index]
        z = _open(shard, self._cache)

        def get(key):
            full = f"{name}|{key}"
            if full not in z.files:
                raise KeyError(f"missing '{full}' in {shard}")
            return z[full]

        content = get("content").astype(np.float32)
        f0_hz = get("f0_interp").astype(np.float32).reshape(-1)
        uv = get("uv").astype(np.float32).reshape(-1)
        loudness = get("loudness").astype(np.float32).reshape(-1)
        mel = get("mel").astype(np.float32)
        frames = mel.shape[1]
        if content.ndim != 2 or content.shape[0] != frames:
            raise ValueError(
                f"{name}: content must be [T,C] aligned to mel T={frames}, got {content.shape}"
            )
        for key, value in (("f0_interp", f0_hz), ("uv", uv), ("loudness", loudness)):
            if len(value) != frames:
                raise ValueError(f"{name}: {key} has {len(value)} frames, expected {frames}")
        if content.shape[1] != self.content_dim:
            raise ValueError(
                f"{name}: content dim {content.shape[1]} does not match dataset {self.content_dim}"
            )

        return {
            "content": content,
            "f0_logf0": np.log2(np.maximum(f0_hz, 1.0)).astype(np.float32),
            "uv": uv,
            "loudness": loudness,
            "target_mel": mel,
            "spk_id": np.int64(self._mapped_id(dbdir, self.spk_map)),
            "style_id": np.int64(self._mapped_id(dbdir, self.style_map)),
            "item_name": name,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def close(self) -> None:
        """Close cached NPZ handles (important for deleting/moving datasets on Windows)."""
        for archive in self._cache.values():
            if archive is not None:
                archive.close()
        self._cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def svc_collate_fn(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_frames = max(item["target_mel"].shape[1] for item in batch)
    mel_dim = batch[0]["target_mel"].shape[0]
    content_dim = batch[0]["content"].shape[1]

    content = torch.zeros(batch_size, max_frames, content_dim)
    mel = torch.zeros(batch_size, mel_dim, max_frames)
    frame_mask = torch.ones(batch_size, max_frames, dtype=torch.bool)

    def frame_values(key):
        values = torch.zeros(batch_size, max_frames)
        for i, item in enumerate(batch):
            value = torch.as_tensor(item[key], dtype=torch.float32)
            values[i, :len(value)] = value
        return values

    for i, item in enumerate(batch):
        frames = item["target_mel"].shape[1]
        content[i, :frames] = torch.as_tensor(item["content"], dtype=torch.float32)
        mel[i, :, :frames] = torch.as_tensor(item["target_mel"], dtype=torch.float32)
        frame_mask[i, :frames] = False

    return {
        "content": content,
        "f0_logf0": frame_values("f0_logf0"),
        "uv": frame_values("uv"),
        "loudness": frame_values("loudness"),
        "target_mel": mel,
        "content_mask": frame_mask.clone(),
        "frame_mask": frame_mask,
        "spk_id": torch.tensor([int(item["spk_id"]) for item in batch], dtype=torch.long),
        "style_id": torch.tensor([int(item["style_id"]) for item in batch], dtype=torch.long),
        "item_name": [item["item_name"] for item in batch],
    }
