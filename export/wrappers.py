"""Export wrappers that turn an HarmonicAcousticModel[MultiSpk] into a single ONNX-traceable
graph with the input contract OpenUTAU / DiffSinger drive.

Two families, chosen by the trained ckpt (the only weight-level split is `use_uv`):
  * WrapperA  DiffSinger-compatible : inputs [tokens, durations, f0 (+ spk_embed)]. Requires a
              use_uv=False, n_styles=0 model. hop=512 (pseudo: ×2 upsample in, pairwise-average
              out) is the default OpenUTAU grid; hop=256 native is also offered.
  * WrapperB  optional-featured     : same core but real v/uv input, hop256 (or free hop).

Both swap two things that don't otherwise export:
  1. the excitation — `HarmonicNoiseExcitationONNX` (real-DFT STFT, no exp2/fft/complex), and
  2. length regulation — the training `LengthRegulator` builds a `new_zeros(1, max_len, H)` with
     a Python-int length, which BAKES the frame count into the graph (frozen frame axis). Here a
     dynamic, ONNX-safe regulator (`_lr_onnx`, a broadcast compare + gather, no searchsorted) is
     used instead, so the exported frame axis is truly dynamic. It is numerically identical to
     `repeat_interleave(durations)` (verified in export/verify.py).

Contract (hard): sum(durations) == len(f0). That is exactly how OpenUTAU calls a DiffSinger
acoustic model, and it makes the length-regulated cond and the excitation the same T.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from leapsinger.mel import F_MAX, F_MIN, N_FFT, N_MELS, SR, WIN_LEN
from leapsinger.models.acoustic_base import up2_linear

from .attention_onnx import patch_attention_for_export
from .excitation_onnx import HarmonicNoiseExcitationONNX


def _lr_onnx(x, durations):
    """ONNX-safe length regulation. x [B,Tp,H], durations [B,Tp] int -> [B,T,H] with
    T = sum(durations). Identical to `x[b].repeat_interleave(durations[b])`:
    frame f belongs to phoneme j = #{ends <= f} (ends = cumsum(dur)), computed by a broadcast
    compare + ReduceSum (no searchsorted, no Python-int shapes -> dynamic Range/Slice)."""
    dur = durations.long()
    ends = torch.cumsum(dur, dim=1)                              # [B,Tp]
    Tp = dur.shape[1]
    total = ends[:, -1].max()                                   # 0-dim tensor = T (dynamic)
    frame = torch.arange(total, device=x.device)                # Range(0,T) -> [T] dynamic
    cmp = (ends.unsqueeze(2) <= frame.view(1, 1, -1))           # [B,Tp,T] bool
    ph = cmp.to(torch.int64).sum(dim=1).clamp(max=Tp - 1)       # [B,T] phoneme index per frame
    idx = ph.unsqueeze(-1).expand(-1, -1, x.shape[2])           # [B,T,H]
    return torch.gather(x, 1, idx)                              # [B,T,H]


class _ExportBase(nn.Module):
    """Shared core: encode -> (speaker inject) -> excitation x0 -> flow.inference (Euler)."""

    def __init__(self, model, num_steps: int, speaker: str = "none", frozen_spk=None):
        super().__init__()
        assert speaker in ("none", "embed", "bake"), speaker
        self.model = model
        self.num_steps = int(num_steps)
        self.speaker = speaker
        self.hidden = model.hidden
        # ONNX-friendly excitation matching the model's excitation params + the repo mel recipe.
        self.exc = HarmonicNoiseExcitationONNX(
            n_harm=model.n_harm, sr=SR, exc_hop=model.exc_hop, scale=model.exc_scale,
            noise_ratio=model.noise_ratio, harm_decay=model.harm_decay,
            n_fft=N_FFT, win=WIN_LEN, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
            deterministic=False,
        )
        if speaker == "bake":
            assert frozen_spk is not None, "speaker='bake' needs a frozen_spk vector [H]"
            self.register_buffer("frozen_spk", torch.as_tensor(frozen_spk, dtype=torch.float32).view(-1))

    def _cond(self, tokens, dur256, f0l256, uv256, spk_embed):
        """Encode to cond [1,H,T256]."""
        m = self.model
        orig_lr = m.length_regulator.forward
        m.length_regulator.forward = lambda x, d, ml=None: _lr_onnx(x, d)     # dynamic frames
        restore_attn = patch_attention_for_export(m)                         # dynamic token axis
        try:
            cond = m._encode(tokens, dur256, f0l256, uv256,
                             None, max_frames=None, spk_id=None,
                             style_id=None, style_mix=None)                   # [1,H,T]
        finally:
            m.length_regulator.forward = orig_lr
            restore_attn()
        if self.speaker == "embed":
            cond = cond + spk_embed.view(1, self.hidden, 1)                   # host-mixed spk vector
        elif self.speaker == "bake":
            cond = cond + self.frozen_spk.view(1, self.hidden, 1)
        return cond

    def _run256(self, tokens, dur256, f0l256, uv256, spk_embed=None):
        m = self.model
        cond = self._cond(tokens, dur256, f0l256, uv256, spk_embed)
        exc = self.exc(f0l256, uv256)                                        # [1,mel,T] ln-mel
        x0 = m.flow._norm(exc)                                               # -> [-1,1]
        mel = m.flow.inference(cond, num_steps=self.num_steps, algorithm="euler", x0_init=x0)
        return mel                                                           # [1,mel,T256]


class AcousticExportWrapperA(_ExportBase):
    """DiffSinger-compatible. forward(tokens, durations, f0 [, spk_embed]) -> mel [1, T, mel_bins].
    (Output is the DiffSinger [B, T, mel_bins] layout: OpenUTAU feeds it to the vocoder unchanged.)

    hop=512 (default): durations/f0 are on the hop-512 grid; internally upsampled ×2 to hop256
    (via `up2_linear`), run, then pairwise-averaged back to hop512 (each hop-512 frame = mean of
    the two hop-256 frames it straddles; matches a true hop-512 mel far better than decimation).
    hop=256: native, no resample."""

    def __init__(self, model, num_steps: int, hop: int = 512, speaker: str = "none", frozen_spk=None):
        super().__init__(model, num_steps, speaker=speaker, frozen_spk=frozen_spk)
        assert getattr(model, "use_uv", True) is False, "WrapperA needs a uv-free model (use_uv=False)"
        assert getattr(model, "style_emb", None) is None, "WrapperA needs n_styles=0"
        assert hop in (256, 512), hop
        self.hop = hop

    def forward(self, tokens, durations, f0, spk_embed=None):
        f0l = torch.log2(f0.clamp(min=1.0))                                  # [1,T]
        if self.hop == 512:
            dur256 = durations * 2
            f0l256 = up2_linear(f0l)                       # [1,2T]
            uv256 = torch.ones_like(f0l256)                                  # use_uv=False -> all voiced
            mel256 = self._run256(tokens, dur256, f0l256, uv256, spk_embed)
            # hop256 -> hop512: average adjacent frame pairs. A directly-computed hop-512
            # frame centers at the midpoint of the even/odd hop-256 pair, so the pair mean
            # (= linear interp to that midpoint) matches a true hop-512 mel far better than
            # decimation (`::2`), which lands half a hop (128 samples) off. The hop-512
            # vocoder consumes this mel directly, so closeness to a true hop-512 mel is the
            # quality that matters. T256 = 2*sum(durations) is always even -> equal halves.
            mel = 0.5 * (mel256[:, :, 0::2] + mel256[:, :, 1::2])            # -> hop512 [1,mel,T]
        else:
            uv = torch.ones_like(f0l)
            mel = self._run256(tokens, durations, f0l, uv, spk_embed)        # [1,mel,T]
        # DiffSinger / OpenUTAU consume mel as [B, T, mel_bins] and feed it straight to the vocoder
        # (the renderer never transposes between acoustic and vocoder). Emit that layout.
        return mel.transpose(1, 2)                                           # [1, T, mel_bins]


class AcousticExportWrapperB(_ExportBase):
    """Optional-featured. forward(tokens, durations, f0, uv [, spk_embed]) -> mel.
    Native hop (256 by default): real v/uv gate. No pseudo-512 resample. A style-bearing model
    runs at its base style (style 0); ONNX style control is not wired (styles are Python-side)."""

    def __init__(self, model, num_steps: int, speaker: str = "none", frozen_spk=None):
        super().__init__(model, num_steps, speaker=speaker, frozen_spk=frozen_spk)
        self.use_uv = getattr(model, "use_uv", True)

    def forward(self, tokens, durations, f0, uv, spk_embed=None):
        f0l = torch.log2(f0.clamp(min=1.0))
        uv_in = uv if self.use_uv else torch.ones_like(f0l)
        return self._run256(tokens, durations, f0l, uv_in, spk_embed)
