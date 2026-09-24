from __future__ import annotations

import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, rope_theta, head_dim, context_length, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")

        self.rope_theta = float(rope_theta)
        self.head_dim = int(head_dim)
        self.context_length = int(context_length)

        # one frequency per coordinate pair, then an angle for every (position, pair)
        pair_ids = torch.arange(0, head_dim, 2, dtype=torch.float64, device=device)
        inv_freq = self.rope_theta ** (-pair_ids / head_dim)
        positions = torch.arange(context_length, dtype=torch.float64, device=device)
        angles = positions[:, None] * inv_freq[None, :]

        # fixed tables, not learned, so buffers rather than parameters
        self.register_buffer("cos_table", angles.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin_table", angles.sin().to(torch.float32), persistent=False)

    def forward(self, x, token_positions):
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"expected final dimension head_dim={self.head_dim}, got {x.shape[-1]}"
            )
        if token_positions.is_floating_point() or token_positions.is_complex():
            raise TypeError(
                f"token_positions must have an integer dtype, got {token_positions.dtype}"
            )
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                f"token_positions has length {token_positions.shape[-1]} "
                f"but sequence length is {x.shape[-2]}"
            )
        if token_positions.numel() > 0:
            lo, hi = int(token_positions.min()), int(token_positions.max())
            if lo < 0 or hi >= self.context_length:
                raise ValueError(
                    f"token_positions outside [0, {self.context_length}): min={lo}, max={hi}"
                )

        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]

        # add singleton dims so the angles broadcast over batch and head axes
        batch_dims = x.ndim - 2
        position_batch_dims = token_positions.ndim - 1
        shape = (
            *token_positions.shape[:-1],
            *((1,) * (batch_dims - position_batch_dims)),
            token_positions.shape[-1],
            self.head_dim // 2,
        )
        cos = cos.reshape(shape)
        sin = sin.reshape(shape)

        # rotate each adjacent pair (x1, x2), (x3, x4), ...
        even = x[..., 0::2]
        odd = x[..., 1::2]
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos

        return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)