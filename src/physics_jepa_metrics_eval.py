"""Shared representation-evaluation battery for Physics-JEPA models and baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.physics_representation_metrics import (
    LinearResonanceProbe,
    LinearResponseProbe,
    evaluate_resonance_probe,
    evaluate_response_probe,
    mine_representative_pairs,
    pair_distances,
    physics_similarity_evaluation,
    random_pairs,
    train_linear_probe,
    within_cross_family_correlation,
)


def load_cached_latents(model_dir: str | Path, split: str) -> dict[str, np.ndarray] | None:
    """Load the cached latents for one split, or ``None`` if unavailable."""
    directory = Path(model_dir)
    names = ("z_geometry", "z_online", "z_target", "z_pred", "geometry", "response", "resonance_targets")
    paths = {name: directory / f"{split}_{name}.npy" for name in names}
    if not all(path.is_file() for path in paths.values()):
        return None
    source_ids_path = directory / f"{split}_source_ids.txt"
    result: dict[str, np.ndarray] = {name: np.load(path) for name, path in paths.items()}
    result["source_ids"] = np.asarray(source_ids_path.read_text(encoding="utf-8").splitlines())
    return result


def _probe_response(name: str, features_val: np.ndarray, responses_val: np.ndarray, features_test: np.ndarray, responses_test: np.ndarray, device: str, probe_epochs: int, seed: int) -> dict[str, object]:
    probe = LinearResponseProbe(features_val.shape[1])
    train_linear_probe(probe, features_val, responses_val.reshape(responses_val.shape[0], -1), epochs=probe_epochs, seed=seed, device=device)
    baseline_mean = responses_val.mean(axis=0, keepdims=True)
    metrics = evaluate_response_probe(probe, features_test, responses_test, baseline_mean, device=device)
    probe.eval()
    with torch.no_grad():
        predictions = probe(torch.as_tensor(features_test, dtype=torch.float32).to(device)).cpu().numpy()
    return {"name": name, "has_data": True, "input_dim": int(features_val.shape[1]), **metrics, "predictions": predictions, "baseline_mean": baseline_mean}


def _probe_resonance(features_val: np.ndarray, targets_val: np.ndarray, features_test: np.ndarray, targets_test: np.ndarray, device: str, probe_epochs: int, seed: int) -> dict[str, object]:
    probe = LinearResonanceProbe(features_val.shape[1], targets_val.shape[1])
    train_linear_probe(probe, features_val, targets_val, epochs=probe_epochs, seed=seed, device=device)
    return evaluate_resonance_probe(probe, features_test, targets_test, device=device)


def evaluate_model_representation(
    model_dir: str | Path,
    subset_root: str | Path = "data/processed/sutd_prcm_30k",
    device: str = "cpu",
    num_pairs: int = 20000,
    num_pairs_family: int = 40000,
    probe_epochs: int = 300,
    seed: int = 7,
    load_only_similarity: bool = False,
) -> dict[str, object]:
    directory = Path(model_dir)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    latent_dim = int(config.get("latent_dim", 32))
    val = load_cached_latents(directory, "val")
    test = load_cached_latents(directory, "test")
    if val is None or test is None:
        raise FileNotFoundError(f"Cached latents missing for {directory}; rerun training with --cache-splits val test")

    result: dict[str, object] = {"latent_dim": latent_dim}
    latent_names = ("z_target", "z_pred", "z_geometry")

    probes: dict[str, object] = {}
    if not load_only_similarity:
        for name in latent_names:
            probes[name] = {
                "response": _probe_response(name, val[name], val["response"], test[name], test["response"], device, probe_epochs, seed),
                "resonance": _probe_resonance(val[name], val["resonance_targets"], test[name], test["resonance_targets"], device, probe_epochs, seed),
            }
        result["probes"] = probes

    similarity: dict[str, object] = {}
    distance_cache: dict[str, object] = {}
    for name in latent_names:
        metrics = physics_similarity_evaluation(test[name], test["geometry"], test["response"], num_pairs=num_pairs, seed=seed)
        similarity[name] = metrics
        first, second = random_pairs(len(test[name]), min(num_pairs, 5000), seed)
        distance_cache[name] = pair_distances(
            test[name], test["geometry"], test["response"], first, second
        )
    result["similarity"] = similarity
    result["pair_distances"] = distance_cache

    if not load_only_similarity:
        result["within_cross_family"] = within_cross_family_correlation(
            test["z_pred"], test["geometry"], test["response"], test["source_ids"], num_pairs=num_pairs_family, seed=seed
        )
        case_a, case_b = mine_representative_pairs(test["geometry"], test["response"], num_pairs=num_pairs, seed=seed)
        result["representative_pairs"] = {"case_a": {key: values[:12] for key, values in case_a.items()}, "case_b": {key: values[:12] for key, values in case_b.items()}}
    return result
