from __future__ import annotations

import numbers
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.dtype("<u2")


def load_token_array(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"token file not found: {path}")

    n_bytes = path.stat().st_size
    if n_bytes % TOKEN_DTYPE.itemsize != 0:
        raise ValueError(
            f"file size {n_bytes} is not a whole number of {TOKEN_DTYPE.itemsize}-byte tokens"
        )

    return np.memmap(path, mode="r", dtype=TOKEN_DTYPE)


def _require_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def get_batch(dataset, batch_size, sequence_length, device, generator):
    _require_positive_int("batch_size", batch_size)
    _require_positive_int("sequence_length", sequence_length)
    if getattr(dataset, "ndim", 1) != 1:
        raise ValueError("dataset must be a one-dimensional token array")

    n_tokens = len(dataset)
    if n_tokens < sequence_length + 1:
        raise ValueError(
            f"dataset has {n_tokens} tokens, need at least sequence_length + 1 = {sequence_length + 1}"
        )

    starts = torch.randint(0, n_tokens - sequence_length, (batch_size,), generator=generator)
    windows = np.stack([dataset[start : start + sequence_length + 1] for start in starts.tolist()])
    windows = torch.from_numpy(windows.astype(np.int64)).to(device)
    return windows[:, :-1], windows[:, 1:]
