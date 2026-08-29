"""ONNX-safe self-attention for export.

`torch.nn.MultiheadAttention` does not export with a dynamic sequence length on the legacy
TorchScript exporter: its internal reshape bakes the trace-time token count (you get a runtime
`Reshape` error like `requested shape:{24,2,128}` when a later phrase has a different Np). This
is the same reason DiffSinger ships its own attention rather than `nn.MultiheadAttention`.

`mha_forward_onnx` reproduces `nn.MultiheadAttention(batch_first=True)` self-attention EXACTLY —
same `in_proj_weight/bias`, same `out_proj`, same 1/sqrt(head_dim) scaling — but reshapes with
`-1` for the sequence dim, so the exported token axis stays dynamic. Dropout is identity at
eval. `patch_attention_for_export(model)` swaps it onto every MultiheadAttention instance and
returns a restore() thunk (used as a context, like the `_lr_onnx` swap in wrappers.py).
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def mha_forward_onnx(mha: nn.MultiheadAttention, query, key=None, value=None,
                     key_padding_mask=None, need_weights=False, attn_mask=None, **kw):
    """Self-attention only (query is used for q/k/v — matches FFTBlock's attn(x,x,x)).
    key_padding_mask is ignored: export traces a single unpadded phrase. Returns (out, None)."""
    x = query                                            # [B, L, H]  (batch_first)
    H = mha.embed_dim
    nh = mha.num_heads
    hd = H // nh
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)          # [B, L, 3H]
    q, k, v = qkv.chunk(3, dim=-1)                                   # each [B, L, H]
    B = x.shape[0]
    # reshape with -1 for the (dynamic) sequence dim -> [B, nh, L, hd]
    q = q.reshape(B, -1, nh, hd).transpose(1, 2)
    k = k.reshape(B, -1, nh, hd).transpose(1, 2)
    v = v.reshape(B, -1, nh, hd).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)     # [B, nh, L, L]
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)                                      # [B, nh, L, hd]
    out = out.transpose(1, 2).reshape(B, -1, H)                      # [B, L, H]
    out = mha.out_proj(out)                                          # [B, L, H]
    return out, None


def patch_attention_for_export(model):
    """Swap every nn.MultiheadAttention.forward -> mha_forward_onnx. Returns restore()."""
    patched = []
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            patched.append((m, m.forward))
            m.forward = types.MethodType(mha_forward_onnx, m)

    def restore():
        for m, fwd in patched:
            m.forward = fwd
    return restore
