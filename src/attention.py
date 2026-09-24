from __future__ import annotations

import math

import torch


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