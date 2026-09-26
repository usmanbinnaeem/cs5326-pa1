from __future__ import annotations

import argparse
import json
import numbers
from pathlib import Path

import torch

from src.attention import softmax
from src.checkpoint import load_model_state
from src.train import add_model_arguments, build_model, resolve_device


def generate(
    model,
    prompt_ids,
    max_new_tokens,
    context_length,
    *,
    temperature=1.0,
    top_p=1.0,
    eot_token_id=None,
    generator=None,
):
    if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 1 or prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must be a non-empty one-dimensional tensor")
    if prompt_ids.dtype != torch.long:
        raise TypeError(f"prompt_ids must be torch.long, got {prompt_ids.dtype}")
    model_device = next(model.parameters()).device
    if prompt_ids.device != model_device:
        raise ValueError(f"prompt_ids is on {prompt_ids.device} but the model is on {model_device}")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, numbers.Integral) or max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be a non-negative integer, got {max_new_tokens!r}")
    if not context_length > 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if getattr(model, "context_length", None) != context_length:
        raise ValueError(
            f"context_length={context_length} must equal model.context_length="
            f"{getattr(model, 'context_length', None)}"
        )

    if max_new_tokens == 0:
        return prompt_ids

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            sequence = prompt_ids
            for _ in range(max_new_tokens):
                window = sequence[-context_length:]
                logits = model(window.unsqueeze(0))[0, -1]
                # sample on CPU so a CPU torch.Generator works with any model device
                probs = softmax(logits.float().cpu() / temperature, dim=-1)

                sorted_probs, sorted_ids = torch.sort(probs, descending=True)
                if top_p < 1.0:
                    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
                    sorted_probs = sorted_probs * (mass_before < top_p)
                    sorted_probs = sorted_probs / sorted_probs.sum()

                rank = torch.multinomial(sorted_probs, 1, generator=generator)
                next_id = sorted_ids[rank].to(sequence.device)
                sequence = torch.cat((sequence, next_id))
                if eot_token_id is not None and int(next_id) == eot_token_id:
                    break
        return sequence
    finally:
        model.train(was_training)


def main(argv=None):
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description="Sample from a trained PA1 Transformer LM.")
    parser.add_argument("--weights", default="final_model.pt", help="final_model.pt or a training checkpoint")
    parser.add_argument("--tokenizer", default="data/tinystories/tokenizer/tokenizer.json")
    parser.add_argument("--prompt", action="append", help="repeat for several prompts")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, nargs="+", default=[1.0])
    parser.add_argument("--top-p", type=float, nargs="+", default=[1.0])
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eot-token-id", type=int, default=0)
    parser.add_argument("--no-eot-stop", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="optional JSON file to store every sample")
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    prompts = args.prompt or ["Once upon a time"]

    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = build_model(args, device)
    model.load_state_dict(load_model_state(args.weights))

    results = []
    for prompt in prompts:
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False).ids, dtype=torch.long, device=device
        )
        for temperature in args.temperature:
            for top_p in args.top_p:
                for sample_index in range(args.num_samples):
                    generator = torch.Generator().manual_seed(args.seed + sample_index)
                    ids = generate(
                        model,
                        prompt_ids,
                        args.max_new_tokens,
                        model.context_length,
                        temperature=temperature,
                        top_p=top_p,
                        eot_token_id=None if args.no_eot_stop else args.eot_token_id,
                        generator=generator,
                    ).tolist()
                    text = tokenizer.decode(ids, skip_special_tokens=False)
                    results.append(
                        {
                            "prompt": prompt,
                            "temperature": temperature,
                            "top_p": top_p,
                            "seed": args.seed + sample_index,
                            "num_new_tokens": len(ids) - len(prompt_ids),
                            "text": text,
                        }
                    )
                    print(f"=== T={temperature} top_p={top_p} seed={args.seed + sample_index}")
                    print(text, end="\n\n", flush=True)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
