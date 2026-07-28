from __future__ import annotations

import torch
from torch import Tensor, nn


class AutoregressiveCodePrior(nn.Module):
    """A small GRU prior over flattened VQ-VAE token maps."""

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.bos_index = codebook_size
        self.embedding = nn.Embedding(codebook_size + 1, embedding_dim)
        self.recurrent = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, codebook_size)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("Code prior expects tokens with shape [batch, length]")
        bos = torch.full(
            (tokens.shape[0], 1),
            self.bos_index,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        teacher_inputs = torch.cat([bos, tokens[:, :-1]], dim=1)
        hidden, _state = self.recurrent(self.embedding(teacher_inputs))
        return self.output(hidden)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        sequence_length: int,
        *,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        device = device or next(self.parameters()).device
        current = torch.full(
            (num_samples, 1),
            self.bos_index,
            dtype=torch.long,
            device=device,
        )
        state = None
        generated: list[Tensor] = []
        for _position in range(sequence_length):
            hidden, state = self.recurrent(self.embedding(current), state)
            logits = self.output(hidden[:, -1]) / temperature
            token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            generated.append(token)
            current = token
        return torch.cat(generated, dim=1)

