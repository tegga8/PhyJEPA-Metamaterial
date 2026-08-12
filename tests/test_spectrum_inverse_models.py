import torch

from src.spectrum_inverse_models import SpectrumToGeometryDirectMLP, SpectrumToGeometryLatent


def test_inverse_models_accept_response_only_and_have_expected_shapes():
    response = torch.randn(2, 4, 1001)
    latent_model = SpectrumToGeometryLatent(hidden_dim=32)
    direct_model = SpectrumToGeometryDirectMLP(hidden_dim=32)
    assert latent_model(response).shape == (2, 64, 8, 8)
    assert direct_model(response).shape == (2, 1, 16, 16)


def test_inverse_models_reject_geometry_leakage_shape():
    model = SpectrumToGeometryLatent(hidden_dim=32)
    with torch.no_grad():
        try:
            model(torch.randn(2, 1, 16, 16))
        except ValueError:
            pass
        else:
            raise AssertionError("inverse model accepted geometry-shaped input")
