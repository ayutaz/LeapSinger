"""DiffWave-style gated residual block used by the mel rectified-flow velocity backbone."""

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# DiffWave bidirectional gated residual block: dilated conv -> gated activation -> residual/skip
#   - output 2xch (filter|gate) -> tanh x sigmoid gated activation
#   - non-causal (bidirectional): no WaveNet-style causal mask
#   - cond (condition) and step (diffusion step) are added in. The conditioner
#     side has no step/cond (cond_dim=0).
# ----------------------------------------------------------------------------
class GatedResBlock(nn.Module):
    def __init__(self, ch: int, kernel: int = 3, dilation: int = 1,
                 cond_dim: int = 0, step_dim: int = 0):
        super().__init__()
        pad = dilation * (kernel - 1) // 2  # 'same' convolution (odd kernel; pad=dilation for kernel=3)
        self.dconv = nn.Conv1d(ch, 2 * ch, kernel, padding=pad, dilation=dilation)
        self.cond_proj = nn.Conv1d(cond_dim, 2 * ch, 1) if cond_dim > 0 else None
        self.step_proj = nn.Linear(step_dim, 2 * ch) if step_dim > 0 else None
        self.res_conv = nn.Conv1d(ch, ch, 1)
        self.skip_conv = nn.Conv1d(ch, ch, 1)

    def forward(self, x, cond=None, step_emb=None):
        h = self.dconv(x)                                   # [B, 2C, T]
        if self.step_proj is not None and step_emb is not None:
            h = h + self.step_proj(step_emb).unsqueeze(-1)  # step injection (broadcast over time)
        if self.cond_proj is not None and cond is not None:
            h = h + self.cond_proj(cond)                    # condition injection (per frame)
        a, b = h.chunk(2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)                # gated activation
        res = (x + self.res_conv(h)) * (2.0 ** -0.5)        # residual (DiffWave-style 1/sqrt(2) scale)
        skip = self.skip_conv(h)                            # skip
        return res, skip
