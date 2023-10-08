from typing import Optional

import torch
from torch import cfloat
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module

from einops import rearrange, repeat, reduce

# helpers

def exists(v):
    return v is not None

# complex attention - Eilers et al
# https://arxiv.org/abs/2306.09827

def eilers_complex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None
):
    """
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

# complex attention - Yang et al
# https://arxiv.org/abs/1910.10202

def yang_complex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None
):
    """
    section 3.2 equation 3
    """
    device = q.device

    assert all([t.dtype == cfloat for t in (q, k, v)])
    q, k, v = [torch.view_as_real(t) for t in (q, k, v)]

    q = repeat(q, 'b h n d c -> (r1 c r2) b h n d', r1 = 2, r2 = 2)
    k = repeat(k, 'b h n d c -> (r c) b h n d', r = 4)
    v = repeat(v, 'b h n d c -> (r c) b h n d', r = 4)

    scale = q.shape[-1] ** -0.5

    sim = einsum('... i d, ... j d -> ... i j', q, k)
    sim = sim * scale

    dtype = sim.dtype

    if exists(mask):
        mask = rearrange(mask, 'b j -> 1 b 1 1 j')
        sim = sim.masked_fill(~mask, -torch.finfo(dtype).max)

    attn = sim.softmax(dim = -1)

    o = einsum('c ... i j, c ... j d -> ... i d c', attn, v)

    sign = torch.tensor([
        1., -1., -1., -1.,   # real component
        1.,  1.,  1., -1.    # imag component
    ], dtype = dtype, device = device)

    o = reduce(o * sign, 'b h n d (c r) -> b h n d c', 'sum', c = 2)

    return torch.view_as_complex(o)
