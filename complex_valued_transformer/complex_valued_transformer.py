from typing import Optional

import torch
from torch import cfloat
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module

from einops import rearrange, repeat

# helpers

def exists(v):
    return v is not None

# complex attention

def complex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None
):
    """
    https://arxiv.org/abs/2306.09827

    section 4.1 equation 8
    """

    assert all([t.dtype == cfloat for t in (q, k, v)])

    scale = q.shape[-1] ** -0.5

    # following eq 4

    q = rearrange(q, 'b h i d -> b h i 1 d')
    k = rearrange(k, 'b h j d -> b h 1 j d')

    qk_phase_diff = q.angle() - k.angle()

    sim = (q.abs() * k.abs() * torch.cos(qk_phase_diff)).sum(dim = -1)
    sim = sim * scale

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

    attn = sim.softmax(dim = -1)

    attn = attn + 0j
    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out
