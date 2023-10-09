from typing import Optional
from functools import partial

import torch
from torch import cfloat
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# complex attention
# https://arxiv.org/abs/2306.09827

def complex_attention_real(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    causal = False
):
    """
    section 4.1 equation 8
    """

    assert all([t.dtype == cfloat for t in (q, k, v)])
    q, k, v = [torch.view_as_real(t) for t in (q, k, v)]
    q, k, v = map(lambda t: rearrange(t, '... d c -> ... (d c)'), (q, k, v))

    scale = q.shape[-1] ** -0.5

    # following eq 4

    sim = einsum('b h i d, b h j d -> b h i j', q, k)
    sim = sim * scale

    dtype = sim.dtype

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, -torch.finfo(dtype).max)

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(dtype).max)

    attn = sim.softmax(dim = -1)

    out = einsum('b h i j, b h j d -> b h i d', attn, v)

    out = rearrange(out, '... (d c) -> ... d c', c = 2)
    return torch.view_as_complex(out)

# complex attention - Yang et al
# https://arxiv.org/abs/1910.10202

def complex_attention_complete(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    causal = False
):
    """
    section 3.2 equation 3
    """
    batch, device = q.shape[0], q.device

    assert all([t.dtype == cfloat for t in (q, k, v)])
    q, k, v = [torch.view_as_real(t) for t in (q, k, v)]

    q = repeat(q, 'b h n d c -> (r1 c r2 b) h n d', r1 = 2, r2 = 2)
    k = repeat(k, 'b h n d c -> (r c b) h n d', r = 4)
    v = repeat(v, 'b h n d c -> (r c b) h n d', r = 4)

    if exists(mask):
        mask = repeat(mask, 'b ... -> (r b) ...', r = 8)

    scale = q.shape[-1] ** -0.5

    sim = einsum('... i d, ... j d -> ... i j', q, k)
    sim = sim * scale

    dtype = sim.dtype

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, -torch.finfo(dtype).max)

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(dtype).max)

    attn = sim.softmax(dim = -1)

    o = einsum('... i j, ... j d -> ... i d', attn, v)

    o = rearrange(o, '(r c b) ... -> b ... (c r)', r = 8, b = batch)

    sign = torch.tensor([
        1., -1., -1., -1.,   # real component
        1.,  1.,  1., -1.    # imag component
    ], dtype = dtype, device = device)

    o = reduce(o * sign, 'b h n d (c r) -> b h n d c', 'sum', c = 2)

    return torch.view_as_complex(o)

# complex multihead attention

class ComplexMultiheadAttention(Module):
    def __init__(
        self,
        dim,
        *,
        causal = False,
        dim_head = 32,
        heads = 8,
        complete_complex = False # whether to use complete complex formulation (Yang et al.) or just the real component, which reduces down to usual dot product on real and imaginary components flattened into the feature dimension
    ):
        super().__init__()
        dim_inner = heads * dim_head

        self.to_q = nn.Linear(dim, dim_inner, bias = False, dtype = cfloat)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias = False, dtype = cfloat)
        self.to_out = nn.Linear(dim_inner, dim, bias = False, dtype = cfloat)

        attend = complex_attention_complete if complete_complex else complex_attention_real
        self.attend = partial(attend, causal = causal)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(self, x, context = None, mask = None):
        has_context = exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(self.split_heads, (q, k, v))

        o = self.attend(q, k, v, mask = mask)

        o = self.merge_heads(o)
        return self.to_out(o)

# rmsnorm

class ComplexRMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(dim, dtype = cfloat))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# feedforward with mod relu
# https://arxiv.org/abs/1511.06464v4

class ModReLU(Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.bias) * torch.exp(1.j * torch.angle(x))


def ComplexFeedForward(dim, mult = 4):
    dim_inner = dim * mult
    return nn.Sequential(
        nn.Linear(dim, dim_inner, dtype = cfloat),
        ModReLU(),
        nn.Linear(dim_inner, dim, dtype = cfloat)
    )

# complex transformer

class ComplexTransformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        num_tokens: Optional[int] = None,
        causal = False,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        complete_complex = False
    ):
        super().__init__()

        self.has_embed = exists(num_tokens)

        if exists(num_tokens):
            self.embed = nn.Parameter(torch.randn((num_tokens, dim), dtype = cfloat))

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                ComplexRMSNorm(dim),
                ComplexMultiheadAttention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, complete_complex = complete_complex),
                ComplexRMSNorm(dim),
                ComplexFeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = ComplexRMSNorm(dim)

        self.to_logits = nn.Linear(dim, num_tokens, dtype = cfloat)

    def forward(self, x):

        if self.has_embed:
            x = self.embed[x]

        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x)) + x
            x = ff(ff_norm(x)) + x

        x = self.norm(x)

        if not self.has_embed:
            return x

        logits = self.to_logits(x)
        return logits
