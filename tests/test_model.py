import torch

from src.models import ForwardSurrogateCNN, ResponseAwareSurrogateCNN


def test_forward_surrogate_shape_and_gradients():
    model = ForwardSurrogateCNN()
    geometry = torch.rand(3, 1, 16, 16)
    prediction = model(geometry)
    assert prediction.shape == (3, 4, 1001)
    prediction.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_response_aware_surrogate_shape_and_gradients():
    model = ResponseAwareSurrogateCNN()
    geometry = torch.rand(2, 1, 16, 16, requires_grad=True)
    prediction = model(geometry)
    assert prediction.shape == (2, 4, 1001)
    prediction.square().mean().backward()
    assert torch.isfinite(geometry.grad).all()
    assert torch.count_nonzero(geometry.grad) > 0
