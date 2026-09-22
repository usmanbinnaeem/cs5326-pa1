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


def get_batch(dataset, batch_size, sequence_length, device, generator):
    n_tokens = len(dataset)
    if n_tokens < sequence_length + 1:
        raise ValueError("dataset is too small for the requested sequence length")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    starts = torch.randint(0, n_tokens - sequence_length, size=(batch_size,), device=device, generator=generator)
    x = torch.stack([torch.tensor(dataset[start:start + sequence_length], device=device) for start in starts])
    y = torch.stack([torch.tensor(dataset[start + 1:start + sequence_length + 1], device=device) for start in starts])
    return x, y