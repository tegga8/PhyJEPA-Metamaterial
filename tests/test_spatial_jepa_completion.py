import torch

from src.spatial_jepa_completion_losses import spatial_jepa_loss, spatial_latent_statistics
from src.spatial_jepa_completion_model import SpatialJEPACompletionModel, compose_binary_spatial_completion


def test_spatial_shapes_are_preserved():
    model = SpatialJEPACompletionModel()
    outputs = model(torch.rand(3, 2, 16, 16), torch.rand(3, 1, 16, 16))
    assert outputs["z_context"].shape == (3, 64, 8, 8)
    assert outputs["z_target"].shape == (3, 64, 8, 8)
    assert outputs["z_pred"].shape == (3, 64, 8, 8)
    assert outputs["logits"].shape == (3, 1, 16, 16)
    assert outputs["z_context"].ndim == 4


def test_target_encoder_is_ema_only_and_has_no_grads():
    model = SpatialJEPACompletionModel()
    trainable_ids = {id(parameter) for parameter in model.trainable_parameters()}
    assert all(id(parameter) not in trainable_ids for parameter in model.target_encoder.parameters())
    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    model.update_target_encoder()
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_spatial_jepa_loss_has_context_predictor_gradients_but_not_target():
    predicted = torch.randn(2, 64, 8, 8, requires_grad=True)
    target = torch.randn(2, 64, 8, 8, requires_grad=True)
    loss = spatial_jepa_loss(predicted, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert target.grad is None


def test_spatial_decoder_gradients_are_nonzero():
    model = SpatialJEPACompletionModel()
    outputs = model(torch.rand(2, 2, 16, 16), torch.rand(2, 1, 16, 16))
    loss = outputs["logits"].square().mean()
    loss.backward()
    for parameter in (*model.context_encoder.parameters(), *model.predictor.parameters(), *model.decoder.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_mask_channel_changes_context_representation():
    model = SpatialJEPACompletionModel()
    geometry = torch.zeros(1, 1, 16, 16)
    first = model.context_encoder(torch.cat((geometry, torch.zeros_like(geometry)), dim=1))
    second = model.context_encoder(torch.cat((geometry, torch.ones_like(geometry)), dim=1))
    assert not torch.allclose(first, second)


def test_spatial_latent_statistics_are_finite():
    values = spatial_latent_statistics(torch.randn(8, 64, 8, 8), torch.randn(8, 64, 8, 8), torch.randn(8, 64, 8, 8))
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())


def test_known_pixels_are_preserved_exactly():
    inputs = torch.zeros(1, 2, 16, 16)
    inputs[:, 0, 0, 1] = 1
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 1, 0] = 1
    inputs[:, 1:2] = mask
    probabilities = torch.zeros(1, 1, 16, 16)
    probabilities[:, :, 1, 0] = 1
    completed = compose_binary_spatial_completion(probabilities, inputs, mask)
    assert torch.equal(completed[:, :, 0, 1], torch.ones(1, 1))
    assert completed[:, :, 1, 0].item() == 1
