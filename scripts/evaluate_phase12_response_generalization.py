"""Phase 12 diagnostic evaluation for the fixed Phase 10 generator.

This script deliberately does not train an inverse model.  It reuses the
Phase 10 generator/candidate cache and creates only the response-space stress
test targets requested for Phase 12.  The learned forward CNN is always named
the *screening surrogate* in artifacts produced here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_phase10_stochastic_inverse_design import (
    complexity_group,
    is_valid,
    nearest_latent_mse,
    topology,
    validity_limits,
)
from scripts.train_phase9_inverse_baselines import InverseDataset, load_forward, load_geometry_autoencoder, nearest_neighbor_predictions, resolve_device, response_mse, set_seed
from src.conditional_latent_vae import ConditionalLatentVAE
from src.spatial_jepa_completion_model import SpatialJEPACompletionModel


GROUPS = ("simple", "medium", "complex")
PAIR_TYPES = ("simple-simple", "medium-medium", "complex-complex", "simple-medium", "simple-complex", "medium-complex")


def sha1(geometry: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(geometry.astype(np.uint8)).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def safe_median(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(values)) if values else None


def hamming_upper(geometries: np.ndarray) -> np.ndarray:
    rows, columns = np.triu_indices(len(geometries), k=1)
    return np.mean(geometries[rows] != geometries[columns], axis=(1, 2, 3)) if len(rows) else np.empty(0, dtype=np.float32)


def response_upper(responses: np.ndarray) -> np.ndarray:
    rows, columns = np.triu_indices(len(responses), k=1)
    return np.mean(np.square(responses[rows] - responses[columns]), axis=(1, 2)) if len(rows) else np.empty(0, dtype=np.float32)


def load_phase10_rows(phase10_dir: Path, candidates: np.ndarray) -> list[dict[str, Any]]:
    with (phase10_dir / "candidate_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["target_index"] = int(row["target_index"])
        row["candidate_index"] = int(row["candidate_index"])
        row["valid"] = row["valid"].strip().lower() == "true"
        for key in row:
            if key not in {"target_sample_id", "complexity_group", "valid", "target_index", "candidate_index"}:
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
        geometry = candidates[row["target_index"], row["candidate_index"], 0]
        row["geometry_hash"] = sha1(geometry)
        row["generation_source"] = "phase10_cached"
        row["noise_protocol"] = "Phase 10 torch.Generator(seed=1042) sequence; cache reused"
    return rows


def encode_latents(autoencoder: torch.nn.Module, geometries: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(geometries), batch_size):
            tensor = torch.from_numpy(geometries[start:start + batch_size].astype(np.float32)).to(device)
            values.append(autoencoder.encode(tensor).cpu().numpy())
    return np.concatenate(values, axis=0)


def screen_geometries(forward: torch.nn.Module, geometries: np.ndarray, target_normalized_30k: np.ndarray, device: torch.device, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    predicted, errors = [], []
    with torch.inference_mode():
        for start in range(0, len(geometries), batch_size):
            binary = torch.from_numpy(geometries[start:start + batch_size].astype(np.float32)).to(device)
            response = forward(binary)
            target = torch.from_numpy(target_normalized_30k[start:start + len(binary)]).to(device)
            predicted.append(response.cpu().numpy())
            errors.append(response_mse(response, target).cpu().numpy())
    return np.concatenate(predicted), np.concatenate(errors)


def construct_interpolations(test: InverseDataset, groups: list[str], pairs_per_type: int, seed: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    rng = np.random.default_rng(seed)
    by_group = {group: np.asarray([index for index, name in enumerate(groups) if name == group], dtype=int) for group in GROUPS}
    rows: list[dict[str, Any]] = []
    targets: list[np.ndarray] = []
    for pair_type in PAIR_TYPES:
        first_group, second_group = pair_type.split("-")
        first = by_group[first_group]
        second = by_group[second_group]
        count = min(pairs_per_type, len(first) if first_group == second_group else min(len(first), len(second)))
        if first_group == second_group:
            selected = rng.choice(first, size=(count, 2), replace=False)
        else:
            selected = np.column_stack((rng.choice(first, size=count, replace=False), rng.choice(second, size=count, replace=False)))
        for pair_number, (a, b) in enumerate(selected):
            if test[int(a)]["sample_id"] == test[int(b)]["sample_id"]:
                raise AssertionError("Interpolated source IDs must differ")
            response_a = test[int(a)]["response_raw"].numpy()
            response_b = test[int(b)]["response_raw"].numpy()
            for alpha in (0.25, 0.50, 0.75):
                target_id = f"interp_{pair_type}_{pair_number:02d}_a{int(alpha * 100):02d}"
                targets.append((alpha * response_a + (1.0 - alpha) * response_b).astype(np.float32))
                rows.append({"target_id": target_id, "target_kind": "interpolated_stress_test", "pair_type": pair_type, "source_a_index": int(a), "source_a_id": test[int(a)]["sample_id"], "source_a_complexity": first_group, "source_b_index": int(b), "source_b_id": test[int(b)]["sample_id"], "source_b_complexity": second_group, "alpha": alpha, "construction": "alpha*S_A + (1-alpha)*S_B", "generator_inputs": "normalized_target_response_only"})
    return rows, np.stack(targets)


def generate_interpolated(
    generator: ConditionalLatentVAE,
    autoencoder: torch.nn.Module,
    forward: torch.nn.Module,
    targets_raw: np.ndarray,
    response_mean_5k: np.ndarray,
    response_std_5k: np.ndarray,
    response_mean_30k: np.ndarray,
    response_std_30k: np.ndarray,
    limits: dict[str, float],
    device: torch.device,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    normalized_5k = (targets_raw - response_mean_5k) / response_std_5k
    normalized_30k = (targets_raw - response_mean_30k) / response_std_30k
    count = len(targets_raw)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    candidates = np.empty((count, k, 1, 16, 16), dtype=np.uint8)
    with torch.inference_mode():
        for target_index in range(count):
            response = torch.from_numpy(normalized_5k[target_index:target_index + 1]).to(device)
            response = response.repeat(k, 1, 1)
            noise = torch.randn((k, generator.latent_dim), generator=rng, dtype=torch.float32).to(device)
            latent = generator.sample(response, noise).view(k, 64, 8, 8)
            logits = autoencoder.decode(latent)
            candidates[target_index] = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.uint8)
    flattened = candidates.reshape(count * k, 1, 16, 16)
    target_repeated = np.repeat(normalized_30k, k, axis=0)
    predicted, errors = screen_geometries(forward, flattened, target_repeated, device)
    predicted = predicted.reshape(count, k, 4, 1001)
    errors = errors.reshape(count, k)
    rows = []
    for target_index in range(count):
        for candidate_index in range(k):
            geometry = candidates[target_index, candidate_index, 0]
            valid, topo = is_valid(geometry, limits, True)
            rows.append({"target_index": target_index, "candidate_index": candidate_index, "seed": seed, "geometry_hash": sha1(geometry), "valid": valid, "response_mse": float(errors[target_index, candidate_index]), **topo})
    return candidates, predicted, errors, rows


def pairwise_metrics(geometries: np.ndarray, latents: np.ndarray, responses: np.ndarray, errors: np.ndarray, good: np.ndarray) -> dict[str, float | int | None]:
    dg = hamming_upper(geometries)
    dr = response_upper(responses)
    latent_pairs = hamming_upper(np.zeros((0, 1, 1, 1), dtype=np.uint8))  # shape-safe empty default
    if len(geometries) >= 2:
        i, j = np.triu_indices(len(geometries), k=1)
        latent_pairs = np.mean(np.square(latents[i] - latents[j]), axis=(1, 2, 3))
    good_indices = np.flatnonzero(good)
    good_hamming = hamming_upper(geometries[good_indices]) if len(good_indices) >= 2 else np.empty(0)
    return {
        "candidate_count": int(len(geometries)),
        "good_candidate_count": int(good.sum()),
        "mean_pairwise_geometry_hamming": safe_mean(dg.tolist()),
        "max_pairwise_geometry_hamming": float(dg.max()) if len(dg) else None,
        "mean_pairwise_geometry_latent_mse": safe_mean(latent_pairs.tolist()),
        "mean_pairwise_response_mse": safe_mean(dr.tolist()),
        "mean_good_pairwise_geometry_hamming": safe_mean(good_hamming.tolist()),
        "max_good_pairwise_geometry_hamming": float(good_hamming.max()) if len(good_hamming) else None,
        "response_error_std": float(np.std(errors)),
    }


def make_plots(plot_dir: Path, diversity_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], interpolation_rows: list[dict[str, Any]], jepa_rows: list[dict[str, Any]]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter([r["mean_pairwise_geometry_hamming"] for r in diversity_rows], [r["best_response_error"] for r in diversity_rows], s=10, alpha=.6)
    axis.set(xlabel="mean pairwise geometry Hamming distance", ylabel="best screening MSE")
    figure.tight_layout(); figure.savefig(plot_dir / "response_vs_diversity.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter([r["geometry_distance_from_best"] for r in diversity_rows if r.get("geometry_distance_from_best") is not None], [r["response_error_difference_from_best"] for r in diversity_rows if r.get("geometry_distance_from_best") is not None], s=6, alpha=.35)
    axis.set(xlabel="geometry Hamming from best candidate", ylabel="screening-MSE difference from best")
    figure.tight_layout(); figure.savefig(plot_dir / "geometry_distance_vs_response_distance.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4))
    labels = list(GROUPS); nn = [safe_mean([r["nn_mse"] for r in comparison_rows if r["complexity_group"] == g]) for g in labels]; gen = [safe_mean([r["generator_mse"] for r in comparison_rows if r["complexity_group"] == g]) for g in labels]
    x = np.arange(len(labels)); axis.bar(x-.18, nn, .36, label="train-only NN"); axis.bar(x+.18, gen, .36, label="generator best-of-8"); axis.set_xticks(x, labels); axis.set_ylabel("screening MSE"); axis.legend()
    figure.tight_layout(); figure.savefig(plot_dir / "generator_vs_nn.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist([r["relative_improvement"] for r in comparison_rows], bins=30, color="#4c78a8"); axis.axvline(0, color="black", linewidth=.8); axis.set(xlabel="1 - generator MSE / NN MSE", ylabel="targets")
    figure.tight_layout(); figure.savefig(plot_dir / "relative_improvement.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4))
    values = [safe_mean([r["generator_mse"] for r in comparison_rows if r["complexity_group"] == g]) for g in GROUPS]; axis.bar(GROUPS, values); axis.set_ylabel("generator best-of-8 screening MSE")
    figure.tight_layout(); figure.savefig(plot_dir / "complexity_breakdown.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist([r["good_candidate_count"] for r in diversity_rows], bins=np.arange(10)-.5, rwidth=.8); axis.set(xlabel="good candidates (screening MSE < 0.30)", ylabel="targets")
    figure.tight_layout(); figure.savefig(plot_dir / "useful_diversity.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4))
    labels = [r["representation"] for r in jepa_rows]; iou = [r.get("reconstruction_iou") or 0 for r in jepa_rows]; axis.bar(labels, iou); axis.set_ylim(0, 1); axis.set_ylabel("full-geometry reconstruction IoU")
    figure.tight_layout(); figure.savefig(plot_dir / "jepa_vs_ae.png", dpi=180); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--latent-root", type=Path, default=Path("outputs/phase8_geometry_autoencoder/latents"))
    parser.add_argument("--phase10-dir", type=Path, default=Path("outputs/phase10_stochastic_inverse_design"))
    parser.add_argument("--geometry-autoencoder", type=Path, default=Path("outputs/phase8_geometry_autoencoder/best.pt"))
    parser.add_argument("--forward-checkpoint", type=Path, default=Path("outputs/phase2_5/exp_C_30k_resonance/best.pt"))
    parser.add_argument("--forward-subset-root", type=Path, default=Path("data/processed/sutd_prcm_30k"))
    parser.add_argument("--spatial-jepa-checkpoint", type=Path, default=Path("outputs/phase4_1/exp_4_1A/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase12"))
    parser.add_argument("--pairs-per-type", type=int, default=10)
    parser.add_argument("--candidates-per-target", type=int, default=8)
    parser.add_argument("--good-threshold", type=float, default=.30)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=12012)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.candidates_per_target != 8:
        raise ValueError("Phase 12 fixed comparison requires K=8")
    set_seed(args.seed); device = resolve_device(args.device); start = time.perf_counter()
    for relative in ("targets", "generation", "nearest_neighbor", "diversity", "complexity", "jepa_vs_ae", "plots"):
        (args.output_dir / relative).mkdir(parents=True, exist_ok=True)
    datasets = {split: InverseDataset(args.subset_root, split, args.latent_root) for split in ("train", "test")}
    train, test = datasets["train"], datasets["test"]
    autoencoder = load_geometry_autoencoder(args.geometry_autoencoder, device)
    forward_args = argparse.Namespace(forward_checkpoint=args.forward_checkpoint, forward_subset_root=args.forward_subset_root)
    forward, mean30, std30 = load_forward(forward_args, device)
    checkpoint = torch.load(args.phase10_dir / "generator" / "best.pt", map_location=device, weights_only=False)
    generator = ConditionalLatentVAE(hidden_dim=256).to(device); generator.load_state_dict(checkpoint["model_state_dict"]); generator.eval()
    candidates = np.load(args.phase10_dir / "test_candidates_binary.npy").astype(np.uint8)
    cached_responses = np.load(args.phase10_dir / "test_candidate_responses_normalized_30k.npy").astype(np.float32)
    if candidates.shape != (len(test), 8, 1, 16, 16): raise AssertionError(f"Unexpected Phase 10 cache shape {candidates.shape}")
    phase10_rows = load_phase10_rows(args.phase10_dir, candidates)
    test_geometries = np.stack([test[index]["geometry"].numpy() for index in range(len(test))]).astype(np.uint8)
    train_geometries = np.stack([train[index]["geometry"].numpy() for index in range(len(train))]).astype(np.uint8)
    groups = [complexity_group(geometry[0])[1] for geometry in test_geometries]
    limits = validity_limits(train)
    # Geometry latent is used only as an evaluation metric, never as a generator input.
    cached_latents = encode_latents(autoencoder, candidates.reshape(-1, 1, 16, 16), device).reshape(len(test), 8, 64, 8, 8)
    target_manifest = [{"target_index": i, "target_id": test[i]["sample_id"], "target_kind": "held_out_dataset_response", "complexity_group": groups[i], "generator_inputs": "normalized_target_response_only"} for i in range(len(test))]
    write_csv(args.output_dir / "targets" / "target_manifest.csv", target_manifest)
    write_csv(args.output_dir / "generation" / "candidates.csv", phase10_rows)
    write_csv(args.output_dir / "generation" / "candidate_metrics.csv", phase10_rows)

    diversity_rows: list[dict[str, Any]] = []; useful_rows: list[dict[str, Any]] = []
    for target_index in range(len(test)):
        rows = [row for row in phase10_rows if row["target_index"] == target_index]
        errors = np.asarray([row["response_mse"] for row in rows]); good = np.asarray([row["valid"] and row["response_mse"] < args.good_threshold for row in rows])
        metrics = pairwise_metrics(candidates[target_index], cached_latents[target_index], cached_responses[target_index], errors, good)
        best = int(np.argmin(errors))
        base = {"target_index": target_index, "target_id": test[target_index]["sample_id"], "complexity_group": groups[target_index], "best_response_error": float(errors[best]), **metrics}
        diversity_rows.append(base)
        useful_rows.append({"target": test[target_index]["sample_id"], "good_candidate_count": int(good.sum()), "max_diversity": metrics["max_good_pairwise_geometry_hamming"], "mean_diversity": metrics["mean_good_pairwise_geometry_hamming"], "best_response_error": float(errors[best])})
        for candidate_index in range(8):
            diversity_rows.append({"target_index": target_index, "target_id": test[target_index]["sample_id"], "complexity_group": groups[target_index], "candidate_index": candidate_index, "best_response_error": float(errors[best]), "geometry_distance_from_best": float(np.mean(candidates[target_index, candidate_index] != candidates[target_index, best])), "response_error_difference_from_best": float(errors[candidate_index] - errors[best]), "mean_pairwise_geometry_hamming": metrics["mean_pairwise_geometry_hamming"], "good_candidate_count": int(good.sum())})
    summary_diversity = [row for row in diversity_rows if "candidate_count" in row]
    write_csv(args.output_dir / "diversity" / "diversity_metrics.csv", diversity_rows)
    write_csv(args.output_dir / "diversity" / "useful_diversity.csv", useful_rows)

    # Experiment C: fixed train-only retrieval baseline, re-screened on the same learned surrogate.
    nn_geometries, nn_ids = nearest_neighbor_predictions(train, test)
    raw_test = np.stack([test[i]["response_raw"].numpy() for i in range(len(test))])
    normalized30 = (raw_test - mean30.cpu().numpy()) / std30.cpu().numpy()
    _, nn_errors = screen_geometries(forward, nn_geometries.astype(np.uint8), normalized30, device)
    gen_errors = np.asarray([min(row["response_mse"] for row in phase10_rows if row["target_index"] == i) for i in range(len(test))])
    comparison_rows = []
    for i in range(len(test)):
        rel = 1.0 - gen_errors[i] / nn_errors[i] if nn_errors[i] > 0 else float("nan")
        comparison_rows.append({"target_index": i, "target_id": test[i]["sample_id"], "complexity_group": groups[i], "nearest_train_id": nn_ids[i], "nn_mse": float(nn_errors[i]), "generator_mse": float(gen_errors[i]), "relative_improvement": float(rel), "outcome": "tie" if abs(gen_errors[i] - nn_errors[i]) <= args.tie_tolerance else ("generator_win" if gen_errors[i] < nn_errors[i] else "nn_win"), "generator_best_hash": min([r for r in phase10_rows if r["target_index"] == i], key=lambda r: r["response_mse"])["geometry_hash"]})
    write_csv(args.output_dir / "nearest_neighbor" / "metrics.csv", comparison_rows)

    # Experiment A: only these stress-test targets are newly generated.
    interpolation_manifest, interpolation_raw = construct_interpolations(test, groups, args.pairs_per_type, args.seed)
    write_csv(args.output_dir / "targets" / "interpolated_targets.csv", interpolation_manifest)
    cands_i, responses_i, errors_i, interpolated_candidate_rows = generate_interpolated(generator, autoencoder, forward, interpolation_raw, test.mean, test.std, mean30.cpu().numpy(), std30.cpu().numpy(), limits, device, 8, args.seed + 1)
    for row in interpolated_candidate_rows:
        row.update({key: interpolation_manifest[row["target_index"]][key] for key in ("target_id", "pair_type", "alpha")})
    write_csv(args.output_dir / "generation" / "interpolated_candidate_metrics.csv", interpolated_candidate_rows)
    interpolation_rows = []
    for i, meta in enumerate(interpolation_manifest):
        valid = [row["valid"] for row in interpolated_candidate_rows if row["target_index"] == i]
        interpolation_rows.append({"target_id": meta["target_id"], "complexity_pair": meta["pair_type"], "alpha": meta["alpha"], "best_of_8_mse": float(errors_i[i].min()), "median_mse": float(np.median(errors_i[i])), "validity": float(np.mean(valid))})

    # Experiment D2: Phase 8 reconstruction by target complexity.
    with torch.inference_mode():
        decoded = []
        for start_i in range(0, len(test_geometries), 128):
            batch = torch.from_numpy(test_geometries[start_i:start_i + 128].astype(np.float32)).to(device)
            logits = autoencoder(batch)["logits"]
            decoded.append((torch.sigmoid(logits) >= .5).cpu().numpy().astype(np.uint8))
    ae_binary = np.concatenate(decoded)
    ae_rows = []
    for group in GROUPS:
        indexes = [i for i, value in enumerate(groups) if value == group]
        iou = []
        for i in indexes:
            intersection = np.logical_and(ae_binary[i], test_geometries[i]).sum(); union = np.logical_or(ae_binary[i], test_geometries[i]).sum()
            iou.append(intersection / union if union else 1.0)
        ae_rows.append({"section": "D2_geometry_autoencoder", "complexity_group": group, "samples": len(indexes), "reconstruction_iou": safe_mean(iou)})

    # Experiment D3: adaptive, response-local neighborhoods (nearest distance * 1.25, with 0.01 floor).
    train_response = np.stack([train[i]["response"].numpy() for i in range(len(train))]).reshape(len(train), -1)
    test_response = np.stack([test[i]["response"].numpy() for i in range(len(test))]).reshape(len(test), -1)
    train_norm = np.square(train_response).mean(axis=1); multimodal_rows = []
    for i in [index for index, group in enumerate(groups) if group == "complex"]:
        distance = np.square(test_response[i]).mean() + train_norm - 2.0 * (train_response @ test_response[i]) / train_response.shape[1]
        nearest = float(distance.min()); threshold = max(.01, nearest * 1.25); selected = np.flatnonzero(distance <= threshold)
        if len(selected) < 2: selected = np.argsort(distance)[:min(2, len(distance))]
        geo = train_geometries[selected]; pair = hamming_upper(geo)
        multimodal_rows.append({"target_index": i, "target_id": test[i]["sample_id"], "nearest_response_distance": nearest, "response_neighbor_threshold": threshold, "neighbor_count": int(len(selected)), "mean_neighbor_geometry_hamming": safe_mean(pair.tolist()), "max_neighbor_geometry_hamming": float(pair.max()) if len(pair) else 0.0})

    # Experiment D4: surrogate residual by complexity, evaluated on the held-out inverse split.
    predicted_true, surrogate_errors = screen_geometries(forward, test_geometries, normalized30, device)
    forward_rows = [{"section": "D4_forward_surrogate", "target_index": i, "complexity_group": groups[i], "forward_screening_mse": float(surrogate_errors[i])} for i in range(len(test))]
    complexity_rows = ae_rows + forward_rows + multimodal_rows
    write_csv(args.output_dir / "complexity" / "complexity_metrics.csv", complexity_rows)

    # Experiment E classification uses median test-set pairwise scales as documented empirical cutoffs.
    geometry_scale = float(np.median([r["mean_pairwise_geometry_hamming"] for r in summary_diversity])); response_scale = float(np.median([r["mean_pairwise_response_mse"] for r in summary_diversity]))
    type_counts = defaultdict(int)
    for row in summary_diversity:
        high_geo = row["mean_pairwise_geometry_hamming"] >= geometry_scale; high_response = row["mean_pairwise_response_mse"] >= response_scale
        label = ("Type 1" if high_geo and not high_response else "Type 2" if high_geo else "Type 3" if not high_response else "Type 4")
        row["stochasticity_type"] = label; type_counts[label] += 1
    # Rewrite summary values enriched with classification; append simple per-target row instead of disturbing per-candidate file contract.
    write_csv(args.output_dir / "diversity" / "stochasticity_classification.csv", summary_diversity)

    # Experiment F: a valid response-only JEPA comparison is unavailable because the existing model needs masked geometry context.
    jepa_rows: list[dict[str, Any]] = []
    jepa_iou = None
    if args.spatial_jepa_checkpoint.exists():
        config = json.loads((args.spatial_jepa_checkpoint.parent / "config.json").read_text(encoding="utf-8"))
        jepa = SpatialJEPACompletionModel(config["latent_channels"], config["predictor_hidden_channels"], config["ema_decay"]).to(device)
        jepa.load_state_dict(torch.load(args.spatial_jepa_checkpoint, map_location=device, weights_only=False)["model_state_dict"]); jepa.eval()
        values = []
        with torch.inference_mode():
            for start_i in range(0, len(test_geometries), 128):
                x = torch.from_numpy(test_geometries[start_i:start_i + 128].astype(np.float32)).to(device)
                binary = (torch.sigmoid(jepa.decoder(jepa.encode_target(x))) >= .5).cpu().numpy().astype(np.uint8)
                for pred, truth in zip(binary, test_geometries[start_i:start_i + 128]):
                    union = np.logical_or(pred, truth).sum(); values.append(np.logical_and(pred, truth).sum() / union if union else 1.0)
        jepa_iou = safe_mean(values)
    jepa_rows = [
        {"representation": "Phase 8 geometry autoencoder", "reconstruction_iou": safe_mean([r["reconstruction_iou"] for r in ae_rows]), "deterministic_inverse_response_mse": 0.4441315422058105, "best_of_8_response_mse": float(np.mean(gen_errors)), "validity": safe_mean([float(r["valid"]) for r in phase10_rows]), "diversity": safe_mean([r["mean_pairwise_geometry_hamming"] for r in summary_diversity]), "novelty": safe_mean([r["nearest_train_pixel_hamming"] for r in phase10_rows]), "complex_mse": safe_mean([r["generator_mse"] for r in comparison_rows if r["complexity_group"] == "complex"]), "comparison_status": "response-only inverse baseline"},
        {"representation": "validated spatial JEPA completion", "reconstruction_iou": jepa_iou, "deterministic_inverse_response_mse": None, "best_of_8_response_mse": None, "validity": None, "diversity": None, "novelty": None, "complex_mse": None, "comparison_status": "not comparable: inference requires masked geometry context; supplying it would violate Phase 12 anti-leakage"},
    ]
    write_csv(args.output_dir / "jepa_vs_ae" / "comparison.csv", jepa_rows)
    make_plots(args.output_dir / "plots", diversity_rows, comparison_rows, interpolation_rows, jepa_rows)

    aggregate_by_group = {}
    for group in GROUPS:
        selected = [r for r in comparison_rows if r["complexity_group"] == group]
        aggregate_by_group[group] = {"targets": len(selected), "nn_mse_mean": safe_mean([r["nn_mse"] for r in selected]), "generator_mse_mean": safe_mean([r["generator_mse"] for r in selected]), "relative_improvement_mean": safe_mean([r["relative_improvement"] for r in selected]), "generator_win_fraction": safe_mean([float(r["outcome"] == "generator_win") for r in selected])}
    overall = {"nn_mse_mean": safe_mean([r["nn_mse"] for r in comparison_rows]), "nn_mse_median": safe_median([r["nn_mse"] for r in comparison_rows]), "generator_mse_mean": safe_mean([r["generator_mse"] for r in comparison_rows]), "generator_mse_median": safe_median([r["generator_mse"] for r in comparison_rows]), "relative_improvement_mean": safe_mean([r["relative_improvement"] for r in comparison_rows]), "relative_improvement_median": safe_median([r["relative_improvement"] for r in comparison_rows]), "generator_win_fraction": safe_mean([float(r["outcome"] == "generator_win") for r in comparison_rows]), "nn_win_fraction": safe_mean([float(r["outcome"] == "nn_win") for r in comparison_rows]), "tie_fraction": safe_mean([float(r["outcome"] == "tie") for r in comparison_rows])}
    interpolation_summary = {pair: {"targets": len([r for r in interpolation_rows if r["complexity_pair"] == pair]), "best_of_8_mse": safe_mean([r["best_of_8_mse"] for r in interpolation_rows if r["complexity_pair"] == pair]), "validity": safe_mean([r["validity"] for r in interpolation_rows if r["complexity_pair"] == pair])} for pair in PAIR_TYPES}
    d2 = {r["complexity_group"]: r["reconstruction_iou"] for r in ae_rows}; d4 = {group: safe_mean([r["forward_screening_mse"] for r in forward_rows if r["complexity_group"] == group]) for group in GROUPS}
    d3 = {"complex_targets": len(multimodal_rows), "mean_neighbor_count": safe_mean([r["neighbor_count"] for r in multimodal_rows]), "mean_neighbor_geometry_hamming": safe_mean([r["mean_neighbor_geometry_hamming"] for r in multimodal_rows])}
    decision = "E: nearest-neighbor remains substantially better in mean screening MSE; do not claim successful inverse design. Diagnose response-space coverage and conditioning before scaling."
    metrics = {"phase": 12, "objective": "response-space generalization, multimodality, and bottleneck diagnosis", "config": {"seed": args.seed, "K": 8, "good_candidate_threshold": args.good_threshold, "good_threshold_interpretation": "empirical learned-screening threshold, not a physical law", "tie_tolerance": args.tie_tolerance, "device": str(device), "runtime_seconds": time.perf_counter() - start, "phase10_checkpoint": str(args.phase10_dir / "generator" / "best.pt"), "normalization_5k": str(args.subset_root / "train_response_stats.npz"), "normalization_30k": str(args.forward_subset_root / "train_response_stats.npz")}, "anti_leakage": {"generator_inference_inputs": ["normalized_target_response", "random_noise"], "assertion": "No source ID, geometry, mask, complexity, latent, or split label is passed to generator.sample; source IDs are evaluation metadata only."}, "interpolated": {"summary": interpolation_summary, "overall_best_of_8_mse": safe_mean([r["best_of_8_mse"] for r in interpolation_rows]), "overall_validity": safe_mean([r["validity"] for r in interpolation_rows])}, "generator_vs_nearest_neighbor": {"overall": overall, "by_complexity": aggregate_by_group}, "diversity": {"useful_diversity_rate": safe_mean([float(r["good_candidate_count"] >= 2) for r in summary_diversity]), "average_good_candidate_count": safe_mean([r["good_candidate_count"] for r in summary_diversity]), "pairwise_geometry_hamming_mean": safe_mean([r["mean_pairwise_geometry_hamming"] for r in summary_diversity]), "stochasticity_cutoffs": {"geometry_hamming_median": geometry_scale, "response_pairwise_mse_median": response_scale}, "types": {key: value / len(summary_diversity) for key, value in type_counts.items()}}, "complexity_diagnosis": {"D1_em_conditioner": {"architecture": "SpectrumMLPBackbone: Flatten [4,1001] -> Linear(4004,256) -> GELU -> Linear(256,256) -> GELU", "parameter_count": sum(p.numel() for p in generator.response_backbone.parameters()), "latent_feature_dim": 256, "frequency_resolution_before_pooling": 1001, "global_pooling": False, "interpretation": "No explicit pooling discards bins, but flattening provides no frequency-local inductive bias."}, "D2_geometry_autoencoder_iou": d2, "D3_response_local_multimodality": d3, "D4_forward_surrogate_screening_mse": d4}, "jepa_vs_ae": {"comparison": jepa_rows, "interpretation": "JEPA-specific response-only inverse benefit is not established: the validated checkpoint requires geometry context at inference."}, "architecture_decision": decision, "scientific_claims": {"proven": ["The fixed generator produces target-conditioned candidates using only response and noise.", "The train-only nearest-neighbor baseline is stronger on mean learned-screening MSE."], "supported_not_proven": ["Candidate sampling exhibits geometry and surrogate-response variation.", "Complex targets are substantially harder under the learned screening model."], "not_established": ["Generated structures satisfy Maxwell-equation responses.", "JEPA-specific representation learning improves response-only inverse design."]}}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps({**vars(args), "python": platform.python_version(), "torch": torch.__version__}, indent=2, default=str) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", [{"metric": "held_out_generator_best_of_8_mse", "value": overall["generator_mse_mean"]}, {"metric": "held_out_nn_mse", "value": overall["nn_mse_mean"]}, {"metric": "interpolated_best_of_8_mse", "value": metrics["interpolated"]["overall_best_of_8_mse"]}, {"metric": "useful_diversity_rate", "value": metrics["diversity"]["useful_diversity_rate"]}])
    print(json.dumps({"output_dir": str(args.output_dir), "generator_vs_nn": overall, "interpolated": metrics["interpolated"], "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
