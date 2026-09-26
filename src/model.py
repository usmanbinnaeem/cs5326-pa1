from __future__ import annotations

import torch

from src.attention import CausalGroupedQueryAttention
from src.layers import Embedding, Linear, RMSNorm, Swiglu


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_q_heads,
        n_kv_heads,
        d_ff,
        context_length,
        rope_theta,
        norm_eps=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, norm_eps=norm_eps, device=device, dtype=dtype)
        self.attention = CausalGroupedQueryAttention(
            d_model, n_q_heads, n_kv_heads, context_length, rope_theta, device=device, dtype=dtype
        )
        self.ffn_norm = RMSNorm(d_model, norm_eps=norm_eps, device=device, dtype=dtype)
        self.ffn = Swiglu(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attention(self.attention_norm(x), token_positions=token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        n_q_heads,
        n_kv_heads,
        d_ff,
        rope_theta,
        norm_eps=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = torch.nn.ModuleList(
            TransformerBlock(
                d_model,
                n_q_heads,
                n_kv_heads,
                d_ff,
                context_length,
                rope_theta,
                norm_eps=norm_eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(d_model, norm_eps=norm_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids, token_positions=None):
        if token_ids.ndim == 0:
            raise ValueError("token_ids must have a sequence dimension")
        seq_len = token_ids.shape[-1]
        if not 1 <= seq_len <= self.context_length:
            raise ValueError(
                f"sequence length {seq_len} outside [1, context_length={self.context_length}]"
            )

        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions=token_positions)
        return self.lm_head(self.final_norm(x))
