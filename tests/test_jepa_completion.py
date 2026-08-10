import torch

from src.jepa_completion_losses import jepa_loss, latent_variance_metrics
from src.jepa_completion_model import JEPACompletionModel, compose_binary_jepa_completion


def test_jepa_shapes_and_decoder_output():
    model = JEPACompletionModel()
    inputs = torch.rand(3, 2, 16, 16)
    target = torch.rand(3, 1, 16, 16)
    outputs = model(inputs, target)
    assert outputs["z_context"].shape == (3, 128)
    assert outputs["z_target"].shape == (3, 128)
    assert outputs["z_pred"].shape == (3, 128)
    assert outputs["logits"].shape == (3, 1, 16, 16)


def test_target_encoder_has_no_optimizer_parameters_and_ema_changes_it():
    model = JEPACompletionModel()
    trainable_ids = {id(parameter) for parameter in model.trainable_parameters()}
    assert all(id(parameter) not in trainable_ids for parameter in model.target_encoder.parameters())
    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    model.update_target_encoder()
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_jepa_loss_is_finite_and_target_is_stop_gradient():
    predicted = torch.randn(4, 128, requires_grad=True)
    target = torch.randn(4, 128, requires_grad=True)
    loss = jepa_loss(predicted, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert target.grad is None


def test_latent_variance_metrics_are_finite():
    metrics = latent_variance_metrics(torch.randn(8, 128), torch.randn(8, 128), torch.randn(8, 128))
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_known_pixels_are_preserved_by_jepa_compositing():
    inputs = torch.zeros(1, 2, 16, 16)
    inputs[:, 0, 0, 1] = 1
    inputs[:, 1, 1, 0] = 1
    mask = inputs[:, 1:2]
    probabilities = torch.zeros(1, 1, 16, 16)
    probabilities[:, :, 1, 0] = 1
    completed = compose_binary_jepa_completion(probabilities, inputs, mask)
    assert completed[:, :, 0, 1].item() == 1
    assert completed[:, :, 0, 0].item() == 0
    assert completed[:, :, 1, 1].item() == 0
