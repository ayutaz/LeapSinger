"""Offline singing-voice-conversion acoustic model.

The model intentionally consumes precomputed, mel-frame-aligned content
features. ContentVec/HuBERT and pitch extraction therefore stay outside the
training graph, leaving VRAM for the acoustic model and discriminator.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from leapsinger.models.acoustic import HarmonicAcousticModel
from leapsinger.modules.encoders.content_adapter import ContentAdapter


class HarmonicSVCModel(HarmonicAcousticModel):
    """Content + F0/UV + loudness -> target-singer mel.

    This is the offline teacher architecture. The content adapter uses
    non-causal convolutions; a streaming student is a separate later stage.
    ``content_features`` must be ``[B, T, content_dim]`` on the same frame grid
    as F0, loudness, and the target mel.
    """

    def __init__(
        self,
        *,
        content_dim: int,
        content_layers: int = 2,
        content_dropout: float = 0.1,
        n_speakers: int = 0,
        spk_dim: int = 0,
        **kwargs,
    ):
        # Build the proven flow/excitation stack but discard the SVS-only
        # phoneme encoder and length regulator. SVC conditioning is frame based.
        super().__init__(n_phonemes=1, n_speakers=0, **kwargs)
        self.phoneme_encoder = None
        self.length_regulator = None

        self.content_dim = int(content_dim)
        self.content_layers = int(content_layers)
        self.content_dropout = float(content_dropout)
        self.content_adapter = ContentAdapter(
            content_dim, self.hidden, n_layers=content_layers, dropout=content_dropout
        )
        self.loudness_emb = nn.Linear(1, self.hidden)

        self.spk_n = int(n_speakers)
        self.spk_dim = int(spk_dim)
        self.spk_bank = None
        self.spk_proj = None
        if self.spk_n > 0 and self.spk_dim > 0:
            self.spk_bank = nn.Embedding(self.spk_n, self.spk_dim)
            self.spk_proj = nn.Linear(self.spk_dim, self.hidden)
            nn.init.normal_(self.spk_bank.weight, 0, self.spk_dim ** -0.5)
        elif self.spk_n > 0:
            self.spk_emb = nn.Embedding(self.spk_n, self.hidden)
            nn.init.normal_(self.spk_emb.weight, 0, self.hidden ** -0.5)

    def _speaker_condition(self, spk_id: torch.Tensor | None) -> torch.Tensor | None:
        if spk_id is None or self.spk_n == 0:
            return None
        if self.spk_bank is not None:
            return self.spk_proj(self.spk_bank(spk_id))
        return self.spk_emb(spk_id)

    def _encode(
        self,
        content_features: torch.Tensor,
        f0_logf0: torch.Tensor,
        uv: torch.Tensor,
        loudness: torch.Tensor,
        content_mask: torch.Tensor | None = None,
        *,
        max_frames: int | None = None,
        spk_id: torch.Tensor | None = None,
        style_id: torch.Tensor | None = None,
        style_mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if content_features.ndim != 3:
            raise ValueError("content_features must be [B,T,C]")
        batch, frames, _ = content_features.shape
        expected = (batch, frames)
        for name, value in (("f0_logf0", f0_logf0), ("uv", uv), ("loudness", loudness)):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")
        if max_frames is not None and frames != int(max_frames):
            raise ValueError(
                f"content features must be aligned to {max_frames} mel frames, got {frames}"
            )

        c = self.content_adapter(content_features, content_mask)
        c = c + self.f0_emb(f0_logf0.unsqueeze(-1)).transpose(1, 2)
        c = c + self.loudness_emb(loudness.unsqueeze(-1)).transpose(1, 2)
        if self.use_uv:
            c = c + self.uv_emb((uv > 0.5).long()).transpose(1, 2)

        speaker = self._speaker_condition(spk_id)
        if speaker is not None:
            c = c + speaker[:, :, None]
        if self.style_emb is not None:
            if style_mix is not None:
                c = c + (style_mix @ self.style_emb.weight)[:, :, None]
            elif style_id is not None:
                c = c + self.style_emb(style_id)[:, :, None]
        if content_mask is not None:
            c = c.masked_fill(content_mask[:, None, :], 0.0)
        return c

    def forward(
        self,
        content_features: torch.Tensor,
        f0_logf0: torch.Tensor,
        uv: torch.Tensor,
        loudness: torch.Tensor,
        target_mel: torch.Tensor,
        *,
        content_mask: torch.Tensor | None = None,
        spk_id: torch.Tensor | None = None,
        style_id: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        harm_wave: torch.Tensor | None = None,
    ) -> dict:
        cond = self._encode(
            content_features, f0_logf0, uv, loudness, content_mask,
            max_frames=target_mel.shape[2], spk_id=spk_id, style_id=style_id,
        )
        x0_init = self._excitation_x0(f0_logf0, uv, harm_wave=harm_wave)
        flow_mask = None if frame_mask is None else (~frame_mask)
        flow_loss, x1_pred = self.flow.compute_loss(
            target_mel, cond, x0_init=x0_init, mask=flow_mask
        )
        recon = self._recon_loss(x1_pred, target_mel, frame_mask)
        return {
            "flow": flow_loss,
            "recon": recon,
            "total": flow_loss,
            "x1_pred": x1_pred,
            "cond": cond,
        }

    @torch.no_grad()
    def infer(
        self,
        content_features: torch.Tensor,
        f0_logf0: torch.Tensor,
        uv: torch.Tensor,
        loudness: torch.Tensor,
        *,
        n_frames: int | None = None,
        num_steps: int = 1,
        algorithm: str = "euler",
        content_mask: torch.Tensor | None = None,
        spk_id: torch.Tensor | None = None,
        style_id: torch.Tensor | None = None,
        style_mix: torch.Tensor | None = None,
        harm_wave: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frames = content_features.shape[1] if n_frames is None else int(n_frames)
        cond = self._encode(
            content_features, f0_logf0, uv, loudness, content_mask,
            max_frames=frames, spk_id=spk_id, style_id=style_id, style_mix=style_mix,
        )
        x0_init = self._excitation_x0(f0_logf0, uv, harm_wave=harm_wave)
        return self.flow.inference(
            cond, num_steps=num_steps, algorithm=algorithm, x0_init=x0_init
        )
