import torch

from src.models import ForwardSurrogateCNN


def test_forward_surrogate_shape_and_gradients():
    model = ForwardSurrogateCNN()
    geometry = torch.rand(3, 1, 16, 16)
    prediction = model(geometry)
    assert prediction.shape == (3, 4, 1001)
    prediction.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
