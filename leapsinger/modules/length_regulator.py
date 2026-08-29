"""Length regulation: expand phoneme-level features to frame-level."""

import torch
import torch.nn as nn


class LengthRegulator(nn.Module):
    """Expands phoneme-level features to frame-level using per-phoneme durations.

    ベクトル化(cumsum + searchsorted + gather): フレーム→所属音素を一括で求めて gather。
    per-バッチ Python ループ(repeat_interleave)を排し GPU の直列化を無くす。
    ベクトル化(cumsum + searchsorted + gather)で、素朴な repeat_interleave ループ版と数値等価(bit一致)。"""

    def forward(self, x, durations, max_len=None):
        """
        x:         [B, T_ph, H]
        durations: [B, T_ph]  int (frames per phoneme; padded positions should have dur=0)
        max_len:   int or None (pad to this length; defaults to max sum-of-durations in batch)
        returns:   [B, T_frames, H]
        """
        B, Tp, H = x.shape
        dur = durations.long().clamp(min=0)
        ends = dur.cumsum(dim=1)                                       # [B, Tp] 各音素の終端(排他)
        if max_len is None:
            max_len = int(ends[:, -1].max().item()) if B > 0 else 0
        frame = torch.arange(max_len, device=x.device)                # [max_len]
        ph = torch.searchsorted(ends, frame.unsqueeze(0).expand(B, -1).contiguous(),
                                right=True).clamp(max=Tp - 1)          # [B, max_len] 各フレームの所属音素
        out = x.gather(1, ph.unsqueeze(-1).expand(-1, -1, H))         # [B, max_len, H] = x[b, ph]
        valid = frame.unsqueeze(0) < ends[:, -1:]                     # [B, max_len] sum(dur) 内のみ
        return out * valid.unsqueeze(-1).to(out.dtype)               # 総フレーム超は 0（旧ループの zero-pad と一致）
