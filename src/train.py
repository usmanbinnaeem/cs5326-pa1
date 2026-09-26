from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from src.checkpoint import export_model_state, load_checkpoint, save_checkpoint
from src.data import get_batch, load_token_array
from src.model import TransformerLM
from src.optim import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping


def add_model_arguments(parser):
    group = parser.add_argument_group("model")
    group.add_argument("--vocab-size", type=int, default=8192)
    group.add_argument("--context-length", type=int, default=256)
    group.add_argument("--d-model", type=int, default=512)
    group.add_argument("--num-layers", type=int, default=4)
    group.add_argument("--n-q-heads", type=int, default=16)
    group.add_argument("--n-kv-heads", type=int, default=4)
    group.add_argument("--d-ff", type=int, default=1344)
    group.add_argument("--rope-theta", type=float, default=10_000.0)
    group.add_argument("--norm-eps", type=float, default=1e-5)


def build_model(args, device):
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_q_heads=args.n_q_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
        device=device,
    )


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(name, device):
    if name != "auto":
        return name
    if device.type == "cuda":
        return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    return "fp32"


def autocast_context(precision, device):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def evaluate_validation(
    model, tokens, num_batches, batch_size, sequence_length, device, generator, precision="fp32"
):
    was_training = model.training
    model.eval()
    try:
        total = torch.zeros((), device=device)
        with torch.inference_mode():
            for _ in range(num_batches):
                x, y = get_batch(tokens, batch_size, sequence_length, device, generator)
                with autocast_context(precision, device):
                    logits = model(x)
                total += cross_entropy(logits, y)
        return float(total) / num_batches
    finally:
        model.train(was_training)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the PA1 Transformer LM.")
    parser.add_argument("--train-data", default="data/tinystories/data/train.bin")
    parser.add_argument("--val-data", default="data/tinystories/data/validation.bin")
    parser.add_argument("--run-dir", default="runs/final")
    parser.add_argument("--resume", action="store_true", help="continue from <run-dir>/checkpoint.pt")
    add_model_arguments(parser)

    train = parser.add_argument_group("training")
    train.add_argument("--sequence-length", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train.add_argument("--num-steps", type=int, default=10_000)
    train.add_argument("--learning-rate-max", type=float, default=3e-4)
    train.add_argument("--learning-rate-min", type=float, default=3e-5)
    train.add_argument("--warmup-steps", type=int, default=200)
    train.add_argument("--cosine-steps", type=int, default=9_999)
    train.add_argument("--beta1", type=float, default=0.9)
    train.add_argument("--beta2", type=float, default=0.95)
    train.add_argument("--adam-eps", type=float, default=1e-8)
    train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--max-grad-norm", type=float, default=1.0)

    run = parser.add_argument_group("evaluation, logging, checkpointing")
    run.add_argument("--eval-interval", type=int, default=200)
    run.add_argument("--log-interval", type=int, default=10)
    run.add_argument("--checkpoint-interval", type=int, default=500)
    run.add_argument("--num-validation-batches", type=int, default=20)
    run.add_argument("--val-batch-size", type=int, default=None)

    system = parser.add_argument_group("reproducibility and hardware")
    system.add_argument("--seed", type=int, default=0, help="model initialization seed")
    system.add_argument("--train-seed", type=int, default=1)
    system.add_argument("--val-seed", type=int, default=2)
    system.add_argument("--device", default="auto")
    system.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    system.add_argument("--compile", action="store_true")
    system.add_argument(
        "--overfit-single-batch",
        action="store_true",
        help="sanity check: sample one training batch once and optimize it repeatedly",
    )

    args = parser.parse_args(argv)
    for name in ("eval_interval", "log_interval", "checkpoint_interval", "num_validation_batches"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    if args.sequence_length > args.context_length:
        parser.error("--sequence-length cannot exceed --context-length")
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    return args


def truncate_metrics(path, last_step):
    if not path.exists():
        return
    kept = [line for line in path.read_text().splitlines() if line and json.loads(line)["step"] <= last_step]
    path.write_text("".join(line + "\n" for line in kept))


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    metrics_path = run_dir / "metrics.jsonl"
    if checkpoint_path.exists() and not args.resume:
        raise SystemExit(f"{checkpoint_path} exists; pass --resume or choose a different --run-dir")

    train_tokens = load_token_array(args.train_data)
    val_tokens = load_token_array(args.val_data)

    torch.manual_seed(args.seed)
    model = build_model(args, device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate_max,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    train_generator = torch.Generator().manual_seed(args.train_seed)
    val_generator = torch.Generator().manual_seed(args.val_seed)

    next_step = 0
    if args.resume and checkpoint_path.exists():
        next_step = load_checkpoint(checkpoint_path, model, optimizer, train_generator, val_generator)
        truncate_metrics(metrics_path, next_step)
        print(f"resumed from {checkpoint_path} at step {next_step}")
    elif metrics_path.exists():
        metrics_path.unlink()

    config = {**vars(args), "device": str(device), "precision": precision}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    num_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch_size * args.sequence_length * args.gradient_accumulation_steps
    print(
        f"device={device} precision={precision} params={num_params:,} "
        f"tokens/step={tokens_per_step:,} total_tokens={tokens_per_step * args.num_steps:,}"
    )

    forward_model = torch.compile(model) if args.compile else model
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None
    fixed_batch = (
        get_batch(train_tokens, args.batch_size, args.sequence_length, device, train_generator)
        if args.overfit_single_batch
        else None
    )

    last_log_time = time.perf_counter()
    last_log_step = next_step
    for step in range(next_step, args.num_steps):
        model.train()
        lr = get_lr_cosine_schedule(
            step, args.learning_rate_max, args.learning_rate_min, args.warmup_steps, args.cosine_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # fp16 loss scaling skips updates with overflowing gradients; retry so every step is one real update
        while True:
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for _ in range(args.gradient_accumulation_steps):
                if fixed_batch is None:
                    x, y = get_batch(train_tokens, args.batch_size, args.sequence_length, device, train_generator)
                else:
                    x, y = fixed_batch
                with autocast_context(precision, device):
                    logits = forward_model(x)
                microbatch_loss = cross_entropy(logits, y)
                scaled_loss = microbatch_loss / args.gradient_accumulation_steps
                (scaler.scale(scaled_loss) if scaler else scaled_loss).backward()
                loss_sum += microbatch_loss.detach()
            train_loss = float(loss_sum) / args.gradient_accumulation_steps

            if scaler is None:
                grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
                optimizer.step()
                break
            scaler.unscale_(optimizer)
            grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                break
            print(f"step {step + 1}: fp16 overflow, update skipped; retrying with scale {scaler.get_scale():g}")

        completed_steps = step + 1
        final_step = completed_steps == args.num_steps
        should_validate = final_step or completed_steps % args.eval_interval == 0
        should_log = should_validate or completed_steps % args.log_interval == 0
        should_checkpoint = final_step or completed_steps % args.checkpoint_interval == 0

        val_loss = None
        if should_validate:
            val_loss = evaluate_validation(
                model,
                val_tokens,
                args.num_validation_batches,
                args.val_batch_size,
                args.sequence_length,
                device,
                val_generator,
                precision,
            )

        if should_log:
            now = time.perf_counter()
            tokens_per_second = (completed_steps - last_log_step) * tokens_per_step / (now - last_log_time)
            last_log_time, last_log_step = now, completed_steps
            record = {
                "step": completed_steps,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_seen": completed_steps * tokens_per_step,
                "tokens_per_second": tokens_per_second,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            val_text = f" | val {val_loss:.4f} (ppl {math.exp(val_loss):.2f})" if val_loss is not None else ""
            print(
                f"step {completed_steps:>6}/{args.num_steps} | train {train_loss:.4f}{val_text} "
                f"| lr {lr:.3e} | grad_norm {grad_norm:.3f} | {tokens_per_second:,.0f} tok/s",
                flush=True,
            )

        if should_checkpoint:
            tmp_path = checkpoint_path.with_suffix(".pt.tmp")
            save_checkpoint(model, optimizer, completed_steps, train_generator, val_generator, tmp_path)
            os.replace(tmp_path, checkpoint_path)

    export_model_state(model, run_dir / "final_model.pt")
    print(f"exported FP16 model state to {run_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
