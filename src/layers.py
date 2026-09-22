import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, norm_eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )
        self.norm_eps = norm_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.norm_eps)
        result = (x / rms) * self.weight

        return result.to(in_dtype)


class SiLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Swiglu(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w_gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_down = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
