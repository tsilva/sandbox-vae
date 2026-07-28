import torch
import torch.nn.functional as functional

from latent_lab.models.ae import Autoencoder


def test_autoencoder_reduces_tiny_batch_loss() -> None:
    torch.manual_seed(0)
    inputs = torch.rand(8, 1, 8, 8)
    model = Autoencoder((1, 8, 8), latent_dim=8, hidden_dims=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    with torch.no_grad():
        initial = functional.mse_loss(model(inputs).reconstruction, inputs)

    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.mse_loss(model(inputs).reconstruction, inputs)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final = functional.mse_loss(model(inputs).reconstruction, inputs)
    assert final < initial * 0.5

