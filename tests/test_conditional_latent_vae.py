import torch

from scripts.train_phase10_stochastic_inverse_design import nearest_latent_mse
from src.conditional_latent_vae import ConditionalLatentVAE


def test_stochastic_generator_response_and_noise_contract():
    model = ConditionalLatentVAE(hidden_dim=16)
    response = torch.randn(2, 4, 1001)
    noise = torch.randn(2, 64 * 8 * 8)
    first = model.sample(response, noise)
    second = model.sample(response, noise)
    third = model.sample(response, noise + 0.1)
    assert first.shape == (2, 64 * 8 * 8)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert torch.isfinite(first).all()


def test_stochastic_generator_rejects_wrong_noise_shape():
    model = ConditionalLatentVAE(hidden_dim=16)
    response = torch.randn(1, 4, 1001)
    try:
        model.sample(response, torch.randn(1, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong noise shape was accepted")


def test_nearest_latent_mse_uses_training_distance():
    train = torch.stack([torch.zeros(64, 8, 8), torch.ones(64, 8, 8)]).numpy()
    query = torch.stack([torch.zeros(64, 8, 8), torch.full((64, 8, 8), 2.0)]).numpy()
    distances = nearest_latent_mse(query, train, block_size=1)
    assert distances[0] == 0.0
    assert abs(float(distances[1]) - 1.0) < 1e-6
