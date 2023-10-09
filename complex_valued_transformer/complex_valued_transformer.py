from typing import Optional
from functools import partial

import torch
from torch import cfloat
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from complex_valued_transformer.attend import Attend

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# helper tensor functions

def modulate_with_rotation(x, m):
    if m.dtype == cfloat:
        m = m.abs()

    rot = m.cos() + 1.j * m.sin()
    return x * rot

# complex attention
# https://arxiv.org/abs/2306.09827

def complex_attention_real(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attend: Attend,
    mask: Optional[Tensor] = None
):
    """
    section 4.1 equation 8
    """

    assert all([t.dtype == cfloat for t in (q, k, v)])
    q, k, v = map(torch.view_as_real, (q, k, v))
    q, k, v = map(lambda t: rearrange(t, '... d c -> ... (d c)'), (q, k, v))

    o = attend(q, k, v, mask = mask)

    out = rearrange(out, '... (d c) -> ... d c', c = 2)
    return torch.view_as_complex(out)

# complex attention - Yang et al
# https://arxiv.org/abs/1910.10202

def complex_attention_complete(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attend: Attend,
    mask: Optional[Tensor] = None
):
    """
    section 3.2 equation 3
    """
    batch, device = q.shape[0], q.device

    assert all([t.dtype == cfloat for t in (q, k, v)])
    q, k, v = map(torch.view_as_real, (q, k, v))

    # complex attention =    (MH(A, A, A) − MH(A, B, B) − MH(B, A, B) − MH(B, B, A))
    #                     + i(MH(A, A, B) + MH(A, B, A) + MH(B, A, A) − MH(B, B, B))

    q = repeat(q, 'b h n d c -> (c r b) h n d', r = 2)
    k = repeat(k, 'b h n d c -> (r c b) h n d', r = 2)
    v = repeat(v, 'b h n d c -> (r b) h n (d c)', r = 4)

    if exists(mask):
        mask = repeat(mask, 'b ... -> (r b) ...', r = 4)

    o = attend(q, k, v, mask = mask)

    o = rearrange(o, '(r b) ... (d c) -> (r c) b ... d', r = 4, c = 2)

    indices = torch.tensor([0, 3, 5, 6, 1, 2, 4, 7], dtype = torch.long, device = device)

    o = rearrange(o[indices], '(r c) ... -> ... c r', c = 2)

    sign = torch.tensor([
        [1., -1., -1., -1.],   # real component
        [1.,  1.,  1., -1.]    # imag component
    ], dtype = o.dtype, device = device)

    o = (o * sign).sum(dim = -1)

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
        complete_complex = False, # whether to use complete complex formulation (Yang et al.) or just the real component, which reduces down to usual dot product on real and imaginary components flattened into the feature dimension
        flash = False
    ):
        super().__init__()
        dim_inner = heads * dim_head

        self.to_q = nn.Linear(dim, dim_inner, bias = False, dtype = cfloat)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias = False, dtype = cfloat)
        self.to_out = nn.Linear(dim_inner, dim, bias = False, dtype = cfloat)

        maybe_flash_attn = Attend(
            causal = causal,
            heads = heads,
            flash = flash
        )

        complex_attention = complex_attention_complete if complete_complex else complex_attention_real
        self.attend = partial(complex_attention, attend = maybe_flash_attn)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        x,
        context = None,
        mask = None,
        rotary_emb = None
    ):
        has_context = exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(self.split_heads, (q, k, v))

        if exists(rotary_emb):
            q = q * rotary_emb
            k = k * rotary_emb

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
    def __init__(self, relu_squared = False):
        super().__init__()
        self.pow = 2 if relu_squared else 1
        self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        real = F.relu(torch.abs(x) + self.bias) ** self.pow
        imag = torch.exp(1.j * torch.angle(x))
        return real + imag


def ComplexFeedForward(dim, mult = 4, relu_squared = False):
    dim_inner = dim * mult
    return nn.Sequential(
        nn.Linear(dim, dim_inner, dtype = cfloat),
        ModReLU(relu_squared = relu_squared),
        nn.Linear(dim_inner, dim, dtype = cfloat)
    )

# rotary embeddings
# formulated for complex numbers

class RotaryEmbedding(Module):
    def __init__(self, dim, base = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = einsum('i, j -> i j', t, self.inv_freq)
        return torch.cos(freqs) + 1.j * torch.sin(freqs)

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
        relu_squared = True,
        complete_complex = False,
        rotary_emb = True,
        flash_attn = True
    ):
        super().__init__()

        self.has_embed = exists(num_tokens)

        if exists(num_tokens):
            self.embed = nn.Parameter(torch.randn((num_tokens, dim), dtype = cfloat))

        self.rotary_emb = None
        if rotary_emb:
            self.rotary_emb = RotaryEmbedding(dim_head)

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                ComplexRMSNorm(dim),
                ComplexMultiheadAttention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, complete_complex = complete_complex, flash = flash_attn),
                ComplexRMSNorm(dim),
                ComplexFeedForward(dim = dim, mult = ff_mult, relu_squared = relu_squared)
            ]))

        self.norm = ComplexRMSNorm(dim)

        self.to_logits = nn.Linear(dim, num_tokens, dtype = cfloat)

    def forward(self, x, context = None, mask = None):
        if self.has_embed:
            x = self.embed[x]

        seq_len = x.shape[-2]
        rotary_emb = None

        if exists(self.rotary_emb):
            rotary_emb = self.rotary_emb(seq_len)

        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), context = context, mask = mask, rotary_emb = rotary_emb) + x
            x = ff(ff_norm(x)) + x

        x = self.norm(x)

        if not self.has_embed:
            return x

        logits = self.to_logits(x)
        return logits
