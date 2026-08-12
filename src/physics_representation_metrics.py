"""Distance definitions, correlations, probes, and pair analysis for Physics-JEPA.

All distances used by the representation gate are fixed here so no experiment
can silently redefine them: geometry distance is the mean Hamming fraction
over the 256 pixels, EM distance is the MSE over the Phase-2 normalized
``[4, 1001]`` response, and latent distance is Euclidean distance between
L2-normalized latent vectors.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn

from src.forward_analysis import detect_resonance_features

GEO_QUANTILE = 0.9
EM_QUANTILE = 0.1


def family_of(source_id: str) -> str:
    """Extract the geometry family from a ``FAMILY/Data_XXX/XXXXXX`` source id."""
    return source_id.split("/")[0]


def random_pairs(count: int, num_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``num_pairs`` random index pairs without replacement."""
    rng = np.random.default_rng(seed)
    first = rng.integers(0, count, size=num_pairs)
    second = rng.integers(0, count, size=num_pairs)
    valid = first != second
    while not valid.all():
        missing = int((~valid).sum())
        first[~valid] = rng.integers(0, count, size=missing)
        second[~valid] = rng.integers(0, count, size=missing)
        valid = first != second
    return first, second


def geometry_hamming(geometries: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(geometries[first] - geometries[second]).mean(axis=(1, 2, 3))


def response_mse(responses: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.square(responses[first] - responses[second]).mean(axis=(1, 2))


def latent_l2(latents: np.ndarray, first: np.ndarray, second: np.ndarray, normalize: bool = True) -> np.ndarray:
    values = latents if not normalize else latents / np.linalg.norm(latents, axis=1, keepdims=True)
    return np.linalg.norm(values[first] - values[second], axis=1)


def pair_distances(latent: np.ndarray, geometries: np.ndarray, responses: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "d_geometry": geometry_hamming(geometries, first, second),
        "d_response": response_mse(responses, first, second),
        "d_latent": latent_l2(latent, first, second),
    }


def correlation(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    if method == "pearson":
        value, _ = pearsonr(x, y)
    else:
        value, _ = spearmanr(x, y)
    return float(value) if np.isfinite(value) else 0.0


def correlation_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "spearman": correlation(x, y, "spearman"),
        "pearson": correlation(x, y, "pearson"),
    }


def physics_similarity_evaluation(
    latent: np.ndarray,
    geometries: np.ndarray,
    responses: np.ndarray,
    num_pairs: int = 20000,
    seed: int = 7,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Core experiment: does latent distance reflect EM distance more than geometry?"""
    first, second = random_pairs(len(latent), num_pairs, seed)
    distances = pair_distances(latent, geometries, responses, first, second)
    rho_em = correlation_metrics(distances["d_latent"], distances["d_response"])
    rho_geometry = correlation_metrics(distances["d_latent"], distances["d_geometry"])
    selectivity = abs(rho_em["spearman"]) / (abs(rho_geometry["spearman"]) + eps)
    return {
        "num_pairs": int(len(first)),
        "latent_vs_response_spearman": rho_em["spearman"],
        "latent_vs_response_pearson": rho_em["pearson"],
        "latent_vs_geometry_spearman": rho_geometry["spearman"],
        "latent_vs_geometry_pearson": rho_geometry["pearson"],
        "selectivity_spearman": selectivity,
    }


def within_cross_family_correlation(
    latent: np.ndarray,
    geometries: np.ndarray,
    responses: np.ndarray,
    source_ids: list[str],
    num_pairs: int = 40000,
    seed: int = 7,
) -> dict[str, float]:
    first, second = random_pairs(len(latent), num_pairs, seed)
    families_first = np.asarray([family_of(source_ids[index]) for index in first])
    families_second = np.asarray([family_of(source_ids[index]) for index in second])
    within = families_first == families_second
    cross = ~within
    result: dict[str, float] = {"num_pairs": int(len(first))}
    if within.any():
        distances = pair_distances(latent, geometries, responses, first[within], second[within])
        result["within_family_latent_vs_response_spearman"] = correlation(distances["d_latent"], distances["d_response"])
        result["within_family_latent_vs_geometry_spearman"] = correlation(distances["d_latent"], distances["d_geometry"])
    if cross.any():
        distances = pair_distances(latent, geometries, responses, first[cross], second[cross])
        result["cross_family_latent_vs_response_spearman"] = correlation(distances["d_latent"], distances["d_response"])
        result["cross_family_latent_vs_geometry_spearman"] = correlation(distances["d_latent"], distances["d_geometry"])
    return result


def mine_representative_pairs(
    geometries: np.ndarray,
    responses: np.ndarray,
    num_pairs: int = 20000,
    seed: int = 7,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Find Case A (similar response, different geometry) and Case B (inverse) pairs."""
    first, second = random_pairs(len(geometries), num_pairs, seed)
    d_geometry = geometry_hamming(geometries, first, second)
    d_response = response_mse(responses, first, second)
    geo_high = d_geometry >= np.quantile(d_geometry, GEO_QUANTILE)
    geo_low = d_geometry <= np.quantile(d_geometry, 1.0 - GEO_QUANTILE)
    em_low = d_response <= np.quantile(d_response, EM_QUANTILE)
    em_high = d_response >= np.quantile(d_response, 1.0 - EM_QUANTILE)
    case_a = (geo_high & em_low).nonzero()[0]
    case_b = (geo_low & em_high).nonzero()[0]
    result_a = {"first": first[case_a], "second": second[case_a], "d_geometry": d_geometry[case_a], "d_response": d_response[case_a]}
    result_b = {"first": first[case_b], "second": second[case_b], "d_geometry": d_geometry[case_b], "d_response": d_response[case_b]}
    return result_a, result_b


def resonance_targets(responses_unnormalized: np.ndarray, frequency_ghz: np.ndarray, num_features: int = 3, prominence: float = 0.03) -> np.ndarray:
    """Build fixed-length resonance probe targets ``[N, 2 * (2*num_features) + 2]``.

    Layout: Ty frequencies (3), Ty prominences (3), Rx frequencies (3),
    Rx prominences (3), Ty feature count (1), Rx feature count (1).
    Missing entries are filled with zero and excluded from evaluation.
    """
    responses = np.asarray(responses_unnormalized, dtype=np.float64)
    if responses.ndim != 3 or responses.shape[1:] != (4, 1001):
        raise ValueError(f"Expected unnormalized responses [N, 4, 1001], got {tuple(responses.shape)}")
    output = np.zeros((responses.shape[0], 2 * (2 * num_features) + 2), dtype=np.float32)
    for index in range(responses.shape[0]):
        magnitudes = (np.hypot(responses[index, 0], responses[index, 1]), np.hypot(responses[index, 2], responses[index, 3]))
        for channel, magnitude in enumerate(magnitudes):
            features = detect_resonance_features(magnitude, frequency_ghz, prominence=prominence)
            features = sorted(features, key=lambda feature: -feature["prominence"])[:num_features]
            base = channel * (2 * num_features)
            for rank, feature in enumerate(features):
                output[index, base + rank] = feature["frequency_ghz"]
                output[index, base + num_features + rank] = feature["prominence"]
            output[index, 4 * num_features + channel] = len(features)
    return output


def resonance_probe_metrics(predicted: np.ndarray, target: np.ndarray, num_features: int = 3) -> dict[str, float]:
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    result: dict[str, float] = {}
    for channel, name in ((0, "ty"), (1, "rx")):
        frequencies = target[:, channel * (2 * num_features): channel * (2 * num_features) + num_features]
        predicted_frequencies = predicted[:, channel * (2 * num_features): channel * (2 * num_features) + num_features]
        present = frequencies > 0
        if present.any():
            result[f"{name}_frequency_mae_ghz"] = float(np.abs(predicted_frequencies[present] - frequencies[present]).mean())
        count = target[:, 4 * num_features + channel]
        result[f"{name}_feature_count_mae"] = float(np.abs(predicted[:, 4 * num_features + channel] - count).mean())
    result["feature_count_mae"] = float(np.abs(predicted[:, 4 * num_features:] - target[:, 4 * num_features:]).mean())
    return result


class LinearResponseProbe(nn.Module):
    """Weak linear readout from a latent to the normalized ``[4, 1001]`` response."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Linear(input_dim, 4 * 1001)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent).view(latent.shape[0], 4, 1001)


class LinearResonanceProbe(nn.Module):
    """Weak linear readout from a latent to fixed-size resonance targets."""

    def __init__(self, input_dim: int, output_dim: int = 14) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Linear(input_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


def train_linear_probe(
    probe: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int = 300,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, float]:
    """Fit a weak linear probe; returns training MSE history summary."""
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    targets_tensor = torch.as_tensor(targets, dtype=torch.float32)
    count = features_tensor.shape[0]
    final_mse = float("nan")
    for _ in range(epochs):
        permutation = torch.randperm(count, generator=generator)
        epoch_mse = 0.0
        batches = 0
        for start in range(0, count, batch_size):
            indices = permutation[start:start + batch_size]
            x = features_tensor[indices].to(device)
            y = targets_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(x)
            loss = torch.nn.functional.mse_loss(prediction.reshape(x.shape[0], -1), y)
            loss.backward()
            optimizer.step()
            epoch_mse += float(loss.item())
            batches += 1
        final_mse = epoch_mse / batches
    return {"train_mse": final_mse}


def evaluate_response_probe(
    probe: nn.Module,
    features: np.ndarray,
    responses: np.ndarray,
    baseline_mean: np.ndarray,
    device: str = "cpu",
) -> dict[str, float]:
    probe.eval()
    with torch.no_grad():
        predicted = probe(torch.as_tensor(features, dtype=torch.float32).to(device)).cpu().numpy()
    responses = np.asarray(responses)
    mse = float(np.square(predicted - responses).mean())
    baseline_mse = float(np.square(baseline_mean[None] - responses).mean())
    return {
        "normalized_response_mse": mse,
        "mean_response_baseline_mse": baseline_mse,
        "response_r2_vs_mean": 1.0 - mse / baseline_mse if baseline_mse > 0 else 0.0,
    }


def evaluate_resonance_probe(probe: nn.Module, features: np.ndarray, targets: np.ndarray, device: str = "cpu") -> dict[str, float]:
    probe.eval()
    with torch.no_grad():
        predicted = probe(torch.as_tensor(features, dtype=torch.float32).to(device)).cpu().numpy()
    return resonance_probe_metrics(predicted, np.asarray(targets))
