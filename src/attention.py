from __future__ import annotations

import math

import torch
from einops import rearrange

from src.layers import Linear
from src.rope import RotaryPositionalEmbedding


def softmax(x, dim):
    # subtracting the max doesn't change the result but keeps exp from overflowing
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"query and key feature dimensions differ: {q.shape[-1]} vs {k.shape[-1]}"
        )
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            f"key and value sequence lengths differ: {k.shape[-2]} vs {v.shape[-2]}"
        )

    scores = torch.einsum("...qd,...kd->...qk", q, k) / math.sqrt(q.shape[-1])

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be a boolean tensor, got {mask.dtype}")
        if mask.shape[-2:] != scores.shape[-2:]:
            raise ValueError(
                f"mask must end in (n_q, n_kv) = {tuple(scores.shape[-2:])}, "
                f"got {tuple(mask.shape[-2:])}"
            )
        # a query with every key masked has nothing to attend to
        if not mask.any(dim=-1).all():
            raise ValueError("mask leaves at least one query with no key to attend to")
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, dim=-1)
    return torch.einsum("...qk,...kd->...qd", weights, v)


class CausalGroupedQueryAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_q_heads,
        n_kv_heads,
        context_length,
        rope_theta,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % n_q_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_q_heads={n_q_heads}")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}")

        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_q_heads // n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.context_length = context_length

        self.q_proj = Linear(d_model, n_q_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.out_proj = Linear(n_q_heads * self.head_dim, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, context_length, device=device)

        causal = torch.ones(context_length, context_length, dtype=torch.bool, device=device).tril()
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x, token_positions=None):
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension d_model={self.d_model}, got {x.shape[-1]}")
        seq_len = x.shape[-2]
        if not 1 <= seq_len <= self.context_length:
            raise ValueError(
                f"sequence length {seq_len} outside [1, context_length={self.context_length}]"
            )

        # query head a = h * g + r, so the projected feature axis is laid out as (h, g, d)
        q = rearrange(self.q_proj(x), "... n (h g d) -> ... h g n d", h=self.n_kv_heads, g=self.group_size)
        k = rearrange(self.k_proj(x), "... n (h d) -> ... h n d", h=self.n_kv_heads)
        v = rearrange(self.v_proj(x), "... n (h d) -> ... h n d", h=self.n_kv_heads)

        if token_positions is None:
            positions = torch.arange(seq_len, device=x.device)
            q = self.rope.rotate(q, positions)
            k = self.rope.rotate(k, positions)
        else:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        scores = torch.einsum("...hgid,...hjd->...hgij", q, k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:seq_len, :seq_len], float("-inf"))
        probs = softmax(scores, dim=-1)
        out = torch.einsum("...hgij,...hjd->...hgid", probs, v)

        out = rearrange(out, "... h g n d -> ... n (h g d)")
        return self.out_proj(out)

