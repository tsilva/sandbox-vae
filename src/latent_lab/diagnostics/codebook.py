from __future__ import annotations

import torch
from torch import Tensor


def code_usage(indices: Tensor, codebook_size: int) -> dict[str, Tensor]:
    counts = torch.bincount(indices.reshape(-1), minlength=codebook_size).float()
    probabilities = counts / counts.sum().clamp_min(1)
    perplexity = torch.exp(
        -(probabilities * torch.log(probabilities + 1e-10)).sum()
    )
    return {
        "counts": counts,
        "probabilities": probabilities,
        "perplexity": perplexity,
        "codes_used": (counts > 0).sum(),
    }

