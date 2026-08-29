"""
Phoneme Encoder: FastSpeech2-style Feed-Forward Transformer (FFT).
4 layers, 256d hidden.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTBlock(nn.Module):
    """
    Single Feed-Forward Transformer block (pre-norm).
    Self-attention + Linear FFN (kernel_size=1), as in OpenVPI DiffSinger.
    Using Linear instead of Conv1d keeps parameters at ~0.26M per block (not ~5M).
    """
    def __init__(self, hidden: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn  = nn.MultiheadAttention(hidden, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.ReLU(),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, H]
        r = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = self.dropout(x) + r

        r = x
        x = self.norm2(x)
        x = self.ffn(x)   # Linear: no transpose needed
        return x + r


class PhonemeEncoder(nn.Module):
    """
    Encodes a phoneme ID sequence into hidden representations.

    Args:
        n_phonemes:  vocabulary size (number of distinct phoneme IDs incl. padding=0)
        hidden:      hidden dimension (default 256)
        num_layers:  number of FFT blocks (default 4)
        num_heads:   self-attention heads
        kernel_size: FFN conv kernel size
        dropout:     dropout rate
    """
    def __init__(self, n_phonemes: int, hidden: int = 256,
                 num_layers: int = 4, num_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        # padding_idx は使わない: この音素表では pau=0 が実音素であり、padding_idx=0 にすると
        # pau の埋め込みがゼロに凍結され学習されなくなるため。
        # pad 位置は attention の key_padding_mask と LengthRegulator の dur=0 で無効化
        # されるため、pad の埋め込み値がフレームに届くことはない。
        self.embed = nn.Embedding(n_phonemes, hidden)
        nn.init.normal_(self.embed.weight, mean=0, std=hidden ** -0.5)

        self.pos_scale = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList([
            FFTBlock(hidden, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)

    def _sinusoidal_pos(self, T: int, H: int, device):
        pos  = torch.arange(T, device=device).float().unsqueeze(1)
        dims = torch.arange(0, H, 2, device=device).float()
        freq = torch.exp(-dims * math.log(10000) / H)
        pe   = torch.zeros(T, H, device=device)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        return pe.unsqueeze(0)  # [1, T, H]

    def forward(self, phoneme_ids, padding_mask=None):
        """
        phoneme_ids:  [B, T_ph]   int
        padding_mask: [B, T_ph]   bool, True = padding position
        returns:      [B, T_ph, hidden]
        """
        x = self.embed(phoneme_ids)
        x = x + self.pos_scale * self._sinusoidal_pos(x.shape[1], x.shape[2], x.device)
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)
        return self.norm(x)
