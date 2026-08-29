"""Frame-aligned content features -> LeapSinger conditioning features."""
from __future__ import annotations

import torch
import torch.nn as nn


class ContentAdapter(nn.Module):
    """Adapt precomputed SSL content features to the acoustic-model width.

    Inputs are expected to be aligned to the target mel frame grid before they
    reach the model. Keeping alignment in preprocessing makes mixed-length
    batches deterministic and keeps heavy content encoders out of VRAM during
    acoustic-model training.
    """

    def __init__(self, input_dim: int, hidden: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, kernel_size=1),
            )
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``[B, hidden, T]`` from frame-aligned ``[B, T, input_dim]``."""
        if features.ndim != 3:
            raise ValueError(f"content features must be [B,T,C], got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"content feature dim must be {self.input_dim}, got {features.shape[-1]}"
            )
        if padding_mask is not None and padding_mask.shape != features.shape[:2]:
            raise ValueError(
                f"content padding mask must be {tuple(features.shape[:2])}, "
                f"got {tuple(padding_mask.shape)}"
            )

        x = self.input_proj(self.input_norm(features)).transpose(1, 2)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask[:, None, :], 0.0)
        for block in self.blocks:
            x = x + block(x)
            if padding_mask is not None:
                x = x.masked_fill(padding_mask[:, None, :], 0.0)
        x = self.output_norm(x.transpose(1, 2)).transpose(1, 2)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask[:, None, :], 0.0)
        return x
