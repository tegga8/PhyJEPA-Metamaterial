import torch

from src.completion_model import CompletionCNN


def test_completion_model_shape_probabilities_and_gradients():
    model = CompletionCNN()
    inputs = torch.rand(3, 2, 16, 16, requires_grad=True)
    logits = model(inputs)
    probabilities = model.predict_probabilities(inputs)
    assert logits.shape == (3, 1, 16, 16)
    assert probabilities.shape == (3, 1, 16, 16)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    logits.mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_mask_channel_distinguishes_known_empty_from_unknown():
    model = CompletionCNN()
    known_empty = torch.zeros(1, 2, 16, 16)
    unknown = known_empty.clone()
    unknown[:, 1] = 1.0
    assert not torch.equal(known_empty, unknown)
    assert not torch.allclose(model(known_empty), model(unknown))
    assert model.encoder[0].in_channels == 2
