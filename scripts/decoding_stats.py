"""Quantify the temperature x top-p decoding grid for REPORT.md section 4.

Run from the repository root:  uv run python scripts/decoding_stats.py
Reads report_assets/final_model_samples.json, writes runs/final_3e-3/decoding_stats.json
and report_assets/fig4_decoding_grid.png.

Per (temperature, top_p) cell, averaged over prompts and seeds:
  distinct-2     unique word bigrams / all word bigrams in the completion (diversity)
  repeat-4       share of word 4-grams that repeat an earlier 4-gram (looping)
  non-word rate  share of completion words never seen in the first 20M training
                 tokens (invented / broken words)
  EOT rate       share of samples that ended by emitting <|endoftext|>
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

EOT = "<|endoftext|>"
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
LEXICON_TOKENS = 20_000_000


def words(text):
    return WORD.findall(text.lower())


def build_lexicon():
    tokenizer = Tokenizer.from_file("data/tinystories/tokenizer/tokenizer.json")
    ids = np.memmap("data/tinystories/data/train.bin", dtype="<u2", mode="r")[:LEXICON_TOKENS]
    lexicon = set()
    step = 1_000_000
    for start in range(0, len(ids), step):
        lexicon.update(words(tokenizer.decode(ids[start:start + step].tolist(), skip_special_tokens=True)))
    return lexicon


def sample_stats(completion, lexicon):
    w = words(completion.replace(EOT, " "))
    bigrams = list(zip(w, w[1:]))
    fourgrams = Counter(zip(w, w[1:], w[2:], w[3:]))
    n4 = sum(fourgrams.values())
    return {
        "distinct2": len(set(bigrams)) / max(1, len(bigrams)),
        "repeat4": sum(c - 1 for c in fourgrams.values()) / max(1, n4),
        "nonword": sum(x not in lexicon for x in w) / max(1, len(w)),
        "eot": float(EOT in completion),
        "new_tokens": None,
    }


def main():
    samples = json.loads(Path("report_assets/final_model_samples.json").read_text())
    lexicon = build_lexicon()
    cells = defaultdict(list)
    for s in samples:
        stats = sample_stats(s["text"][len(s["prompt"]):], lexicon)
        stats["new_tokens"] = s["num_new_tokens"]
        cells[(s["temperature"], s["top_p"])].append(stats)

    temps = sorted({t for t, _ in cells})
    tops = sorted({p for _, p in cells})
    keys = ["distinct2", "repeat4", "nonword", "eot", "new_tokens"]
    table = {k: np.array([[np.mean([c[k] for c in cells[(t, p)]]) for p in tops] for t in temps]) for k in keys}

    rows = []
    for i, t in enumerate(temps):
        for j, p in enumerate(tops):
            rows.append({"temperature": t, "top_p": p, "n_samples": len(cells[(t, p)]),
                         **{k: round(float(table[k][i, j]), 4) for k in keys}})
    Path("runs/final_3e-3/decoding_stats.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"lexicon size: {len(lexicon):,} words")
    print(f"{'T':>4} {'p':>4} | {'distinct-2':>10} {'repeat-4':>9} {'non-word':>9} {'EOT':>5} {'new tok':>8}")
    for r in rows:
        print(f"{r['temperature']:>4} {r['top_p']:>4} | {r['distinct2']:>10.3f} {r['repeat4']:>9.3f} "
              f"{r['nonword']:>9.3%} {r['eot']:>5.2f} {r['new_tokens']:>8.1f}")

    panels = [
        ("repeat4", "(a) Repetition: repeated word 4-grams", "{:.1%}", "Oranges"),
        ("distinct2", "(b) Diversity: distinct word bigrams", "{:.2f}", "Blues"),
        ("nonword", "(c) Invented words: not in training text", "{:.1%}", "Reds"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (key, title, fmt, cmap) in zip(axes, panels):
        m = table[key]
        im = ax.imshow(m, cmap=cmap, aspect="auto", vmin=m.min(), vmax=m.max() if m.max() > m.min() else m.min() + 1)
        for i in range(len(temps)):
            for j in range(len(tops)):
                frac = (m[i, j] - m.min()) / max(1e-9, m.max() - m.min())
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if frac > 0.6 else "#0b0b0b")
        ax.set_xticks(range(len(tops)), [f"{p:g}" for p in tops])
        ax.set_yticks(range(len(temps)), [f"{t:g}" for t in temps])
        ax.set_xlabel("top-p", color="#52514e")
        ax.set_ylabel("temperature", color="#52514e")
        ax.set_title(title, loc="left", fontsize=11)
        ax.tick_params(colors="#52514e", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Decoding grid on the final model: 2 prompts x 3 seeds = 6 samples per cell, up to 256 new tokens",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig("report_assets/fig4_decoding_grid.png", dpi=150, facecolor="white")
    print("saved report_assets/fig4_decoding_grid.png")


if __name__ == "__main__":
    main()
