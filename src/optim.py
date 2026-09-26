from __future__ import annotations

import math
import numbers

import torch


def cross_entropy(logits, targets):
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must match logits leading shape "
            f"{tuple(logits.shape[:-1])}"
        )
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    log_normalizer = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


def _validate_hyperparameters(lr, betas, eps, weight_decay):
    if not lr >= 0.0:
        raise ValueError(f"invalid learning rate: {lr}")
    if not eps >= 0.0:
        raise ValueError(f"invalid eps: {eps}")
    if not weight_decay >= 0.0:
        raise ValueError(f"invalid weight_decay: {weight_decay}")
    if len(betas) != 2:
        raise ValueError(f"betas must contain two values, got {betas}")
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"invalid beta1: {beta1}")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"invalid beta2: {beta2}")


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        _validate_hyperparameters(lr, betas, eps, weight_decay)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for group in self.param_groups:
            _validate_hyperparameters(group["lr"], group["betas"], group["eps"], group["weight_decay"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, betas, eps, weight_decay = (
                group["lr"],
                group["betas"],
                group["eps"],
                group["weight_decay"],
            )
            _validate_hyperparameters(lr, betas, eps, weight_decay)
            beta1, beta2 = betas

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                state["step"] += 1
                step = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                param.mul_(1 - lr * weight_decay)
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        return loss


def _require_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")


def get_lr_cosine_schedule(step, learning_rate_max, learning_rate_min, warmup_steps, cosine_steps):
    _require_integer("step", step)
    _require_integer("warmup_steps", warmup_steps)
    _require_integer("cosine_steps", cosine_steps)
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if not 0.0 <= learning_rate_min <= learning_rate_max:
        raise ValueError(
            f"require 0 <= learning_rate_min <= learning_rate_max, got "
            f"{learning_rate_min} and {learning_rate_max}"
        )
    if warmup_steps < 0 or warmup_steps >= cosine_steps:
        raise ValueError(
            f"require 0 <= warmup_steps < cosine_steps, got {warmup_steps} and {cosine_steps}"
        )

    if step < warmup_steps:
        return float(step / warmup_steps * learning_rate_max)
    if step <= cosine_steps:
        progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(learning_rate_min + cosine * (learning_rate_max - learning_rate_min))
    return float(learning_rate_min)


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    if not max_l2_norm > 0:
        raise ValueError(f"max_l2_norm must be positive, got {max_l2_norm}")

    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return 0.0

    total_norm = torch.stack([grad.detach().float().square().sum() for grad in grads]).sum().sqrt()
    total_norm = float(total_norm)
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.detach().mul_(scale)
    return total_norm
