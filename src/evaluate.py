from __future__ import annotations

import argparse
import json
import math

import torch

from src.checkpoint import load_model_state
from src.data import load_token_array
from src.train import add_model_arguments, build_model, evaluate_validation, resolve_device


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standardized PA1 validation (Section 7.3).")
    parser.add_argument("--weights", default="final_model.pt", help="final_model.pt or a training checkpoint")
    parser.add_argument("--val-data", default="data/tinystories/data/validation.bin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="optional JSON file for the result")
    add_model_arguments(parser)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model = build_model(args, device)
    model.load_state_dict(load_model_state(args.weights))

    mean_ce = evaluate_validation(
        model,
        load_token_array(args.val_data),
        args.num_batches,
        args.batch_size,
        args.sequence_length,
        device,
        torch.Generator().manual_seed(args.seed),
    )
    result = {
        "weights": args.weights,
        "seed": args.seed,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "mean_cross_entropy": mean_ce,
        "perplexity": math.exp(mean_ce),
    }
    print(f"validation cross-entropy: {mean_ce:.4f} nats/token")
    print(f"perplexity: {math.exp(mean_ce):.3f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
