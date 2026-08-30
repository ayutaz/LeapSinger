"""Configuration objects, populated from YAML.

Frame / mel-spectrogram settings default to the OpenUTAU DiffSinger convention, with
**hop size 256** as the only deliberate difference — but every field is configurable,
so you can train at any frame size / mel resolution by overriding these in your YAML.
The same MelSpec is shared by preprocessing (mel + F0 frame rate), the dataset loader
(duration → frame expansion), and the model (excitation hop) so they always agree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class MelSpec:
    sr: int = 44_100
    hop: int = 256          # default 256; set 512 (etc.) to train at another frame size
    n_fft: int = 2048
    win: int = 2048
    n_mels: int = 128
    fmin: float = 40.0
    fmax: float = 16_000.0

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop

    @classmethod
    def from_dict(cls, d: dict | None) -> MelSpec:
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)
