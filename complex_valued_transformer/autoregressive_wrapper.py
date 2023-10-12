import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

# helper function

def exists(val):
    return val is not None

def identity(t):
    return t

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# top k filtering

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    probs.scatter_(1, ind, val)
    return probs

class AutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,        
        seq_len,
        pad_value = 0,
        logits_fn = identity
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.net = net
        self.logits_fn = logits_fn

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompt,
        seq_len,
        temperature = 1.0,
        filter_thres = 0.9,
        **kwargs
    ):
        b, t, device = *prompt.shape, prompt.device

        out = prompt

        for _ in range(seq_len):
            logits = self.net(out[:, -self.seq_len:], **kwargs)[:, -1]
            logits = self.logits_fn(logits)

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)

            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim = -1)

        return out[:, t:]

    def forward(self, x, **kwargs):
        x, labels = x[:, :-1], x[:, 1:]
        logits = self.net(x, **kwargs)
        logits = rearrange(self.logits_fn(logits), "b c n -> b n c")
        return F.cross_entropy(logits, labels)
