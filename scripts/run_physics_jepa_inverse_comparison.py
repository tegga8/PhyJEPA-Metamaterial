"""Experiment B: AE representation vs Physics-JEPA representation in the SAME inverse generator.

Controlled representation ablation on the frozen Phase 10 generator contract
(`ConditionalLatentVAE` over the Phase 8 geometry-AE latent, K=8 candidates,
frozen forward screening surrogate ``exp_C_30k_resonance``, 5k subset with the
Phase 10 split 4000/500/500 seed 42).

Two branches, everything identical except the geometry representation used to
define the training target latent:

* Representation A (``ae``): the ordinary Phase 8 geometry-AE latent
  ``z_AE(G)`` (the exact Phase 10 contract, retrained under the new namespace).
* Representation B (``physics_jepa``): the reproduced Physics-JEPA v3
  geometry-derived representation ``z_PJ = E_PJ(G)`` (frozen seed-123 v3
  ``z_pred``) mapped into the AE latent space through a single frozen linear
  least-squares adapter ``A: R^32 -> R^4096`` fit on the training split only
  (the minimum adapter required because the latent shapes differ, documented
  explicitly).  The adapter is used only to build the training target latent
  ``A(z_PJ(G))``; it is never a test-time condition.

Candidate generation uses a shared deterministic noise stream so AE and PJ
receive identical noise per (target, candidate) — a paired comparison.  The
pipeline (training protocol, optimizer, epochs, batch, seeds, K, candidate
filtering, screening model, thresholds) is identical to Phase 10.

No leakage: the generator receives only the normalized target response and
random noise.  Original geometry, masks, source IDs, complexity labels and
geometry latents are never generator inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_phase10_stochastic_inverse_design import (
    complexity_group,
    generate_candidates,
    is_valid,
    topology,
    train as train_generator,
    validity_limits,
)
from scripts.train_phase9_inverse_baselines import (
    InverseDataset,
    load_forward,
    load_geometry_autoencoder,
    nearest_neighbor_predictions,
    resolve_device,
    response_mse,
    set_seed,
)
from scripts.evaluate_phase12_response_generalization import (
    construct_interpolations,
    generate_interpolated,
    screen_geometries,
    write_csv,
)
from src.conditional_latent_vae import ConditionalLatentVAE
from src.physics_jepa import PhysicsJEPAFrequencyRelational

GROUPS = ("simple", "medium", "complex")


class BranchInverseDataset(InverseDataset):
    """Phase 10 contract with a swapped target latent (the only difference)."""

    def __init__(self, root: Path, split: str, latent_root: Path, branch_latents: np.ndarray) -> None:
        super().__init__(root, split, latent_root)
        self.branch_latents = np.asarray(branch_latents, dtype=np.float32)
        if self.branch_latents.shape != self.latents.shape:
            raise ValueError(f"Branch latents {tuple(self.branch_latents.shape)} != AE latents {tuple(self.latents.shape)}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["latent"] = torch.from_numpy(np.asarray(self.branch_latents[index], dtype=np.float32).copy())
        return item


def load_v3(config_path: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = PhysicsJEPAFrequencyRelational(
        latent_dim=int(config["latent_dim"]),
        ema_decay=float(config["ema_decay"]),
        channels=int(config["channels"]),
        token_dim=int(config["token_dim"]),
        num_heads=int(config["num_heads"]),
        num_harmonics=int(config["num_harmonics"]),
        kernel_size=int(config["kernel_size"]),
        dilations=tuple(int(value) for value in config["dilations"]),
        token_stride=int(config["token_stride"]),
        alpha=float(config["loss"]["alpha"]),
        target_centering=bool(config["target_centering"]),
        target_center_decay=float(config["target_center_decay"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def compute_z_pj(model: torch.nn.Module, dataset: InverseDataset, device: torch.device, batch_size: int = 256) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            batch = torch.stack([dataset[index]["geometry"] for index in range(start, stop)]).float().to(device)
            z_geometry = model.geometry_encoder(batch)
            values.append(model.predictor(z_geometry).cpu().numpy())
    return np.concatenate(values, axis=0)


def fit_adapter(z_pj_train: np.ndarray, z_ae_train: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(z_pj_train, z_ae_train, rcond=None)[0]


def adapter_fit_quality(adapter: np.ndarray, z_pj_val: np.ndarray, z_ae_val: np.ndarray) -> dict[str, float]:
    predicted = z_pj_val @ adapter
    residual = predicted - z_ae_val
    sse = float(np.square(residual).sum())
    sst = float(np.square(z_ae_val - z_ae_val.mean(axis=0, keepdims=True)).sum())
    target_variance = float(np.mean(np.var(z_ae_val, axis=0)))
    target_mean_std = float(np.mean(np.sqrt(np.var(z_ae_val, axis=0))))
    return {
        "adapter_r2": 1.0 - sse / sst if sst > 0 else 0.0,
        "target_mean_variance": target_variance,
        "target_mean_std": target_mean_std,
    }


def rank_profile(latent: np.ndarray) -> dict[str, float]:
    values = np.asarray(latent, dtype=np.float64)
    centrality = values - values.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.eigvalsh((centrality.T @ centrality) / (values.shape[0] - 1))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    rank1 = float(eigenvalues[-1] / total) if total > 0 else 0.0
    effective = float(total**2 / np.square(eigenvalues).sum()) if np.square(eigenvalues).sum() > 0 else 0.0
    return {"rank1_fraction": rank1, "effective_rank": effective}


def per_target_summary(rows: list[dict[str, Any]], candidate_binary: np.ndarray, threshold: float, n_candidates: int) -> dict[int, dict[str, Any]]:
    by_target: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(int(row["target_index"]), []).append(row)
    summary: dict[int, dict[str, Any]] = {}
    for target_index, candidates in sorted(by_target.items()):
        errors = np.asarray([float(row["response_mse"]) for row in candidates])
        valid = np.asarray([bool(row["valid"]) for row in candidates])
        good = valid & (errors <= threshold)
        best = int(np.argmin(errors))
        good_geometries = np.asarray(candidate_binary[target_index, :, 0])[good].reshape(-1, 16, 16)
        pairwise = []
        if len(good_geometries) >= 2:
            first, second = np.triu_indices(len(good_geometries), k=1)
            pairwise = np.mean(good_geometries[first] != good_geometries[second], axis=(1, 2)).tolist()
        summary[target_index] = {
            "best_of_8_mse": float(errors[best]),
            "median_of_8_mse": float(np.median(errors)),
            "validity": float(valid.mean()),
            "useful_diversity_count": int(good.sum()),
            "useful_diversity_rate": float(good.sum()) / n_candidates,
            "good_pairwise_geometry_hamming_mean": float(np.mean(pairwise)) if pairwise else None,
            "nearest_train_pixel_hamming_mean": float(np.mean([float(row["nearest_train_pixel_hamming"]) for row in candidates])),
            "nearest_train_latent_mse_mean": float(np.mean([float(row["nearest_train_latent_mse"]) for row in candidates])),
        }
    return summary


def safe_mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def safe_median(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(values)) if values else None


def aggregate(summary: dict[int, dict[str, Any]], target_meta: dict[int, str]) -> dict[str, Any]:
    best = [value["best_of_8_mse"] for value in summary.values()]
    median = [value["median_of_8_mse"] for value in summary.values()]
    validity = [value["validity"] for value in summary.values()]
    diversity = [value["useful_diversity_count"] for value in summary.values()]
    novelty = [value["nearest_train_pixel_hamming_mean"] for value in summary.values()]
    result = {
        "targets": len(summary),
        "best_of_8_mse_mean": safe_mean(best),
        "best_of_8_mse_median": safe_median(best),
        "median_of_8_mse_mean": safe_mean(median),
        "median_of_8_mse_median": safe_median(median),
        "validity_rate": safe_mean(validity),
        "useful_diversity_per_target_mean": safe_mean(diversity),
        "useful_diversity_rate": safe_mean([value["useful_diversity_rate"] for value in summary.values()]),
        "targets_with_useful_diversity_fraction": safe_mean([float(value["useful_diversity_count"] >= 1) for value in summary.values()]),
        "nearest_train_pixel_hamming_mean": safe_mean(novelty),
    }
    for group in GROUPS:
        selected = {index: value for index, value in summary.items() if target_meta[index] == group}
        if not selected:
            continue
        result[group] = {
            "targets": len(selected),
            "best_of_8_mse_mean": safe_mean([value["best_of_8_mse"] for value in selected.values()]),
            "median_of_8_mse_mean": safe_mean([value["median_of_8_mse"] for value in selected.values()]),
            "validity_rate": safe_mean([value["validity"] for value in selected.values()]),
            "useful_diversity_per_target_mean": safe_mean([value["useful_diversity_count"] for value in selected.values()]),
            "nearest_train_pixel_hamming_mean": safe_mean([value["nearest_train_pixel_hamming_mean"] for value in selected.values()]),
            "good_pairwise_geometry_hamming_mean": safe_mean([value["good_pairwise_geometry_hamming_mean"] for value in selected.values()]),
        }
    return result


def plot_pair_comparison(values_a: list[float], values_b: list[float], labels: list[str], title: str, xlabel: str, output: Path) -> Path:
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.scatter(values_a, values_b, s=8, alpha=0.5)
    lo = min(min(values_a), min(values_b))
    hi = max(max(values_a), max(values_b))
    axis.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("AE (Phase 8 representation)")
    axis.set_ylabel("Physics-JEPA representation")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return output if _save_figure(figure, output) else output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _save_figure(figure: plt.Figure, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_complexity_bars(metrics_a: dict[str, Any], metrics_b: dict[str, Any], metric_key: str, ylabel: str, title: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    positions = np.arange(len(GROUPS))
    a = [metrics_a.get(group, {}).get(metric_key) for group in GROUPS]
    b = [metrics_b.get(group, {}).get(metric_key) for group in GROUPS]
    width = 0.38
    axis.bar(positions - width / 2, a, width, label="AE", color="tab:blue")
    axis.bar(positions + width / 2, b, width, label="Physics-JEPA", color="tab:orange")
    axis.set_xticks(positions)
    axis.set_xticklabels(GROUPS)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    _save_figure(figure, output)


def plot_nn_improvement(improvement_a: list[float], improvement_b: list[float], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    lo = min(min(improvement_a), min(improvement_b)) - 0.05
    hi = max(max(improvement_a), max(improvement_b)) + 0.05
    axis.scatter(improvement_a, improvement_b, s=8, alpha=0.5)
    axis.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("I_AE = 1 - E_AE / E_NN")
    axis.set_ylabel("I_PJ = 1 - E_PJ / E_NN")
    axis.set_title("NN-relative improvement: AE vs Physics-JEPA (per target)")
    axis.grid(alpha=0.3)
    _save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--latent-root", type=Path, default=Path("outputs/phase8_geometry_autoencoder/latents"))
    parser.add_argument("--geometry-autoencoder", type=Path, default=Path("outputs/phase8_geometry_autoencoder/best.pt"))
    parser.add_argument("--forward-checkpoint", type=Path, default=Path("outputs/phase2_5/exp_C_30k_resonance/best.pt"))
    parser.add_argument("--forward-subset-root", type=Path, default=Path("data/processed/sutd_prcm_30k"))
    parser.add_argument("--v3-config", type=Path, default=Path("outputs/physics_jepa_v3_repro/correct/config.json"))
    parser.add_argument("--v3-checkpoint", type=Path, default=Path("outputs/physics_jepa_v3_repro/correct/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics_jepa_inverse_comparison"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--beta-kl", type=float, default=1e-4)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--candidates-per-target", type=int, default=8)
    parser.add_argument("--response-threshold", type=float, default=0.30)
    parser.add_argument("--pairs-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.candidates_per_target != 8:
        raise ValueError("Paired comparison requires K=8 (Phase 10 contract)")

    start = time.perf_counter()
    set_seed(args.seed)
    device = resolve_device(args.device)
    root = args.output_dir
    for relative in ("ae", "physics_jepa", "paired_candidates", "plots", "interpolated"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    datasets = {split: InverseDataset(args.subset_root, split, args.latent_root) for split in ("train", "val", "test")}
    train, val, test = datasets["train"], datasets["val"], datasets["test"]
    autoencoder = load_geometry_autoencoder(args.geometry_autoencoder, device)
    forward_args = argparse.Namespace(forward_checkpoint=args.forward_checkpoint, forward_subset_root=args.forward_subset_root)
    forward, forward_mean, forward_std = load_forward(forward_args, device)

    train_geometries = np.stack([train[index]["geometry"].numpy() for index in range(len(train))])
    limits = validity_limits(train)
    test_geometries = np.stack([test[index]["geometry"].numpy() for index in range(len(test))])
    target_groups = [complexity_group(geometry[0])[1] for geometry in test_geometries]
    target_meta = {index: target_groups[index] for index in range(len(test))}

    # ---- Physics-JEPA representation (frozen seed-123 v3 z_pred) ---------------
    v3_model = load_v3(args.v3_config, args.v3_checkpoint, device)
    z_pj = {split: compute_z_pj(v3_model, datasets[split], device) for split in ("train", "val", "test")}

    # ---- Minimum adapter: frozen linear least-squares fit z_PJ -> z_AE ----------
    z_ae_train = np.asarray(train.latents, dtype=np.float32).reshape(len(train), -1)
    adapter = fit_adapter(z_pj["train"], z_ae_train)
    adapter_quality = adapter_fit_quality(adapter, z_pj["val"], np.asarray(val.latents, dtype=np.float32).reshape(len(val), -1))
    branch_b_targets = {split: (z_pj[split] @ adapter).astype(np.float32).reshape(len(datasets[split]), 64, 8, 8) for split in ("train", "val", "test")}
    branch_a_targets = {split: np.asarray(datasets[split].latents, dtype=np.float32).reshape(len(datasets[split]), 64, 8, 8) for split in ("train", "val", "test")}

    # ---- Train both generators identically (only the target latent differs) ------
    training = {}
    generators = {}
    for name, targets in (("ae", branch_a_targets), ("physics_jepa", branch_b_targets)):
        generator = torch.Generator().manual_seed(args.seed)
        branch_datasets = {split: BranchInverseDataset(args.subset_root, split, args.latent_root, targets[split]) for split in ("train", "val", "test")}
        loaders = {
            split: DataLoader(
                dataset,
                batch_size=64,
                shuffle=split == "train",
                generator=generator if split == "train" else None,
                num_workers=0,
            )
            for split, dataset in branch_datasets.items()
        }
        latent_mean_np = np.asarray(targets["train"], dtype=np.float32).mean(axis=0).astype(np.float32)
        latent_std_np = np.asarray(targets["train"], dtype=np.float32).std(axis=0).clip(1e-2).astype(np.float32)
        latent_mean = torch.from_numpy(latent_mean_np).to(device)
        latent_std = torch.from_numpy(latent_std_np).to(device)
        model = ConditionalLatentVAE(args.hidden_dim).to(device)
        output_dir = root / name / "generator"
        result = train_generator(
            model, loaders, autoencoder.decoder, latent_mean, latent_std, device, args.beta_kl, args.prior_weight, args.epochs, args.patience, output_dir
        )
        checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        generators[name] = (model, latent_mean, latent_std)
        training[name] = {"best_epoch": result["best_epoch"], "best_validation_loss": result["best_validation_loss"], "checkpoint": str(output_dir / "best.pt"), "latent_mean_std_from": "train branch targets"}

    # ---- Paired candidate generation (shared noise seed 1042, K=8) ---------------
    generation = {}
    candidates_arrays = {}
    for name in ("ae", "physics_jepa"):
        model, latent_mean, latent_std = generators[name]
        overall, rows, binary, responses = generate_candidates(
            model,
            test,
            autoencoder,
            forward,
            forward_mean,
            forward_std,
            latent_mean,
            latent_std,
            np.asarray(train.latents, dtype=np.float32),
            train_geometries,
            limits,
            device,
            args.candidates_per_target,
            args.seed + 1000,
            args.response_threshold,
        )
        for row in rows:
            row["branch"] = name
        candidates_arrays[name] = binary
        generation[name] = {"overall": overall, "rows": rows, "candidates": binary, "responses": responses}
        write_csv(root / name / "candidate_metrics.csv", rows)

    np.save(root / "paired_candidates" / "ae_candidates_binary.npy", candidates_arrays["ae"])
    np.save(root / "paired_candidates" / "physics_jepa_candidates_binary.npy", candidates_arrays["physics_jepa"])

    # ---- NN baseline (train-only EM nearest neighbor, screened) -------------------
    nn_geometries, nn_ids = nearest_neighbor_predictions(train, test)
    raw_test = np.stack([test[index]["response_raw"].numpy() for index in range(len(test))])
    normalized30 = (raw_test - forward_mean.cpu().numpy()) / forward_std.cpu().numpy()
    _, nn_errors = screen_geometries(forward, nn_geometries.astype(np.uint8), normalized30, device)

    # ---- Per-target aggregation ---------------------------------------------------
    summaries = {name: per_target_summary(generation[name]["rows"], generation[name]["candidates"], args.response_threshold, args.candidates_per_target) for name in ("ae", "physics_jepa")}
    aggregate_metrics = {name: aggregate(summaries[name], target_meta) for name in ("ae", "physics_jepa")}

    comparison_rows = []
    for index in range(len(test)):
        ae = summaries["ae"][index]
        pj = summaries["physics_jepa"][index]
        nn = float(nn_errors[index])
        i_ae = 1.0 - ae["best_of_8_mse"] / nn if nn > 0 else float("nan")
        i_pj = 1.0 - pj["best_of_8_mse"] / nn if nn > 0 else float("nan")
        tolerance = 1e-6
        branch_outcome = "tie" if abs(ae["best_of_8_mse"] - pj["best_of_8_mse"]) <= tolerance else ("physics_jepa_win" if pj["best_of_8_mse"] < ae["best_of_8_mse"] else "ae_win")
        nn_outcome_ae = "tie" if abs(ae["best_of_8_mse"] - nn) <= tolerance else ("ae_win" if ae["best_of_8_mse"] < nn else "nn_win")
        nn_outcome_pj = "tie" if abs(pj["best_of_8_mse"] - nn) <= tolerance else ("physics_jepa_win" if pj["best_of_8_mse"] < nn else "nn_win")
        comparison_rows.append({
            "target_index": index,
            "target_id": test[index]["sample_id"],
            "complexity_group": target_meta[index],
            "nn_mse": nn,
            "nn_nearest_id": nn_ids[index],
            "ae_best_of_8_mse": ae["best_of_8_mse"],
            "ae_median_of_8_mse": ae["median_of_8_mse"],
            "ae_validity": ae["validity"],
            "ae_useful_diversity_count": ae["useful_diversity_count"],
            "ae_novelty": ae["nearest_train_pixel_hamming_mean"],
            "ae_improvement_vs_nn": i_ae,
            "pj_best_of_8_mse": pj["best_of_8_mse"],
            "pj_median_of_8_mse": pj["median_of_8_mse"],
            "pj_validity": pj["validity"],
            "pj_useful_diversity_count": pj["useful_diversity_count"],
            "pj_novelty": pj["nearest_train_pixel_hamming_mean"],
            "pj_improvement_vs_nn": i_pj,
            "branch_outcome": branch_outcome,
            "nn_outcome_ae": nn_outcome_ae,
            "nn_outcome_pj": nn_outcome_pj,
        })
    write_csv(root / "comparison.csv", comparison_rows)

    # ---- Paired summary (fractions of targets where PJ wins/ties/loses) -----------
    wins = [row for row in comparison_rows if row["branch_outcome"] == "physics_jepa_win"]
    ties = [row for row in comparison_rows if row["branch_outcome"] == "tie"]
    losses = [row for row in comparison_rows if row["branch_outcome"] == "ae_win"]
    paired_summary = {
        "targets": len(comparison_rows),
        "physics_jepa_win_fraction": len(wins) / len(comparison_rows),
        "tie_fraction": len(ties) / len(comparison_rows),
        "ae_win_fraction": len(losses) / len(comparison_rows),
        "mean_best_mse_ae": safe_mean([row["ae_best_of_8_mse"] for row in comparison_rows]),
        "mean_best_mse_pj": safe_mean([row["pj_best_of_8_mse"] for row in comparison_rows]),
        "median_best_mse_ae": safe_median([row["ae_best_of_8_mse"] for row in comparison_rows]),
        "median_best_mse_pj": safe_median([row["pj_best_of_8_mse"] for row in comparison_rows]),
        "mean_improvement_ae": safe_mean([row["ae_improvement_vs_nn"] for row in comparison_rows if np.isfinite(row["ae_improvement_vs_nn"])]),
        "median_improvement_ae": safe_median([row["ae_improvement_vs_nn"] for row in comparison_rows if np.isfinite(row["ae_improvement_vs_nn"])]),
        "mean_improvement_pj": safe_mean([row["pj_improvement_vs_nn"] for row in comparison_rows if np.isfinite(row["pj_improvement_vs_nn"])]),
        "median_improvement_pj": safe_median([row["pj_improvement_vs_nn"] for row in comparison_rows if np.isfinite(row["pj_improvement_vs_nn"])]),
        "generator_win_vs_nn_fraction_ae": safe_mean([float(row["nn_outcome_ae"] == "ae_win") for row in comparison_rows]),
        "generator_win_vs_nn_fraction_pj": safe_mean([float(row["nn_outcome_pj"] == "physics_jepa_win") for row in comparison_rows]),
        "nn_win_fraction_ae": safe_mean([float(row["nn_outcome_ae"] == "nn_win") for row in comparison_rows]),
        "nn_win_fraction_pj": safe_mean([float(row["nn_outcome_pj"] == "nn_win") for row in comparison_rows]),
    }

    # ---- Interpolation stress test (same pairs + noise for both branches) ---------
    interpolation_manifest, interpolation_raw = construct_interpolations(test, target_groups, args.pairs_per_type, args.seed)
    interpolation = {}
    for name in ("ae", "physics_jepa"):
        model, latent_mean, latent_std = generators[name]
        _, _, errors_i, rows_i = generate_interpolated(
            model,
            autoencoder,
            forward,
            interpolation_raw,
            test.mean,
            test.std,
            forward_mean.cpu().numpy(),
            forward_std.cpu().numpy(),
            limits,
            device,
            args.candidates_per_target,
            args.seed + 1,
        )
        for row, meta in zip(rows_i, interpolation_manifest):
            row.update({"target_id": meta["target_id"], "pair_type": meta["pair_type"], "alpha": meta["alpha"], "branch": name})
        rows_summary = []
        for i, meta in enumerate(interpolation_manifest):
            valid = [row["valid"] for row in rows_i if row["target_index"] == i]
            rows_summary.append({"target_id": meta["target_id"], "pair_type": meta["pair_type"], "alpha": meta["alpha"], "best_of_8_mse": float(errors_i[i].min()), "median_of_8_mse": float(np.median(errors_i[i])), "validity": float(np.mean(valid))})
        write_csv(root / "interpolated" / f"{name}_candidate_metrics.csv", rows_i)
        interpolation[name] = {
            "best_of_8_mse_mean": safe_mean([row["best_of_8_mse"] for row in rows_summary]),
            "median_of_8_mse_mean": safe_mean([row["median_of_8_mse"] for row in rows_summary]),
            "validity_rate": safe_mean([row["validity"] for row in rows_summary]),
            "by_alpha": {alpha: {"best_of_8_mse_mean": safe_mean([row["best_of_8_mse"] for row in rows_summary if row["alpha"] == alpha]), "validity_rate": safe_mean([row["validity"] for row in rows_summary if row["alpha"] == alpha])} for alpha in (0.25, 0.50, 0.75)},
        }

    # ---- Plots --------------------------------------------------------------------
    plot_dir = root / "plots"
    for source_name, target_name in (("v3_seed_reproduction.png", "v3_seed_reproduction.png"), ("correct_vs_shuffled_seed123.png", "correct_vs_shuffled_seed123.png")):
        source = root.parent / "physics_jepa_v3_repro" / "plots" / source_name
        if source.is_file():
            target = plot_dir / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    plot_pair_comparison(
        [row["ae_best_of_8_mse"] for row in comparison_rows],
        [row["pj_best_of_8_mse"] for row in comparison_rows],
        [row["target_id"] for row in comparison_rows],
        "Best-of-8 screening MSE: AE vs Physics-JEPA",
        "best-of-8 MSE",
        plot_dir / "ae_vs_pj_best_of_8.png",
    )
    plot_pair_comparison(
        [row["ae_median_of_8_mse"] for row in comparison_rows],
        [row["pj_median_of_8_mse"] for row in comparison_rows],
        [row["target_id"] for row in comparison_rows],
        "Median-of-8 screening MSE: AE vs Physics-JEPA",
        "median-of-8 MSE",
        plot_dir / "ae_vs_pj_median.png",
    )
    plot_nn_improvement(
        [row["ae_improvement_vs_nn"] for row in comparison_rows if np.isfinite(row["ae_improvement_vs_nn"])],
        [row["pj_improvement_vs_nn"] for row in comparison_rows if np.isfinite(row["pj_improvement_vs_nn"])],
        plot_dir / "ae_vs_pj_nn_improvement.png",
    )
    plot_complexity_bars(aggregate_metrics["ae"], aggregate_metrics["physics_jepa"], "best_of_8_mse_mean", "best-of-8 MSE (mean)", "Complexity breakdown: best-of-8 screening MSE", plot_dir / "ae_vs_pj_complexity.png")
    plot_complexity_bars(aggregate_metrics["ae"], aggregate_metrics["physics_jepa"], "validity_rate", "validity rate", "Complexity breakdown: candidate validity", plot_dir / "ae_vs_pj_validity.png")
    plot_complexity_bars(aggregate_metrics["ae"], aggregate_metrics["physics_jepa"], "useful_diversity_per_target_mean", "useful candidates / target", "Complexity breakdown: useful diversity (MSE<=0.30)", plot_dir / "ae_vs_pj_diversity.png")

    # ---- Report -------------------------------------------------------------------
    report = {
        "phase": "physics_jepa_vs_ae_inverse_design",
        "objective": "controlled representation ablation inside the same stochastic inverse generator",
        "scientific_question": "Does the Physics-JEPA representation (v3 seed-123 z_pred via frozen linear adapter) improve inverse design vs the ordinary Phase 8 geometry-AE representation?",
        "config": {**vars(args), "python": None, "runtime_seconds": time.perf_counter() - start, "device": str(device)},
        "anti_leakage": {"generator_inference_inputs": ["normalized_target_response", "random_noise"], "representation_usage": "target latent space during training only; the frozen adapter A(z_PJ) defines the training target, never a test-time condition", "original_geometry": False, "partial_geometry": False, "mask": False, "source_id": False, "complexity_label": False, "target_geometry_latent": False, "split_label": False},
        "representations": {
            "ae": {"name": "Phase 8 geometry autoencoder latent z_AE", "checkpoint": str(args.geometry_autoencoder), "target_space": [64, 8, 8], "frozen": True},
            "physics_jepa": {"name": "Physics-JEPA v3 seed-123 z_pred", "checkpoint": str(args.v3_checkpoint), "target_space": [32], "frozen": True, "adapter": "frozen linear least-squares A: R^32 -> R^4096 fit on train split only", "adapter_fit_quality": adapter_quality, "z_pred_rank_profile": rank_profile(z_pj["train"])},
        },
        "generator": {"name": "ConditionalLatentVAE", "hidden_dim": args.hidden_dim, "latent_shape": [64, 8, 8], "beta_kl": args.beta_kl, "prior_weight": args.prior_weight, "epochs": args.epochs, "patience": args.patience, "seed": args.seed, "candidates_per_target": args.candidates_per_target, "noise_pairing": "shared torch.Generator(seed=1042) sequence for both branches"},
        "screening_surrogate": str(args.forward_checkpoint),
        "training": training,
        "nn_baseline": {"name": "train-only EM nearest neighbor", "mean_mse": safe_mean(nn_errors.tolist()), "median_mse": safe_median(nn_errors.tolist())},
        "evaluation": {"threshold_0_30": "learned-screening empirical threshold, not a physical law", "ae": aggregate_metrics["ae"], "physics_jepa": aggregate_metrics["physics_jepa"]},
        "paired_comparison": paired_summary,
        "interpolation": interpolation,
        "decision": None,
    }
    report["config"]["python"] = sys.version.split()[0]
    (root / "metrics.json").write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    summary_rows = [
        ["best_of_8_mse_mean", f"{aggregate_metrics['ae'].get('best_of_8_mse_mean', float('nan')):.6f}", f"{aggregate_metrics['physics_jepa'].get('best_of_8_mse_mean', float('nan')):.6f}"],
        ["best_of_8_mse_median", f"{aggregate_metrics['ae'].get('best_of_8_mse_median', float('nan')):.6f}", f"{aggregate_metrics['physics_jepa'].get('best_of_8_mse_median', float('nan')):.6f}"],
        ["median_of_8_mse_mean", f"{aggregate_metrics['ae'].get('median_of_8_mse_mean', float('nan')):.6f}", f"{aggregate_metrics['physics_jepa'].get('median_of_8_mse_mean', float('nan')):.6f}"],
        ["validity_rate", f"{aggregate_metrics['ae'].get('validity_rate', float('nan')):.4f}", f"{aggregate_metrics['physics_jepa'].get('validity_rate', float('nan')):.4f}"],
        ["useful_diversity_per_target_mean", f"{aggregate_metrics['ae'].get('useful_diversity_per_target_mean', float('nan')):.4f}", f"{aggregate_metrics['physics_jepa'].get('useful_diversity_per_target_mean', float('nan')):.4f}"],
        ["nearest_train_pixel_hamming_mean", f"{aggregate_metrics['ae'].get('nearest_train_pixel_hamming_mean', float('nan')):.4f}", f"{aggregate_metrics['physics_jepa'].get('nearest_train_pixel_hamming_mean', float('nan')):.4f}"],
        ["improvement_vs_nn_mean", f"{paired_summary.get('mean_improvement_ae', float('nan')):.4f}", f"{paired_summary.get('mean_improvement_pj', float('nan')):.4f}"],
        ["physics_jepa_win_fraction", "-", f"{paired_summary.get('physics_jepa_win_fraction', float('nan')):.4f}"],
    ]
    write_csv(root / "summary.csv", [{"metric": row[0], "ae": row[1], "physics_jepa": row[2]} for row in summary_rows])

    print(json.dumps({"output_dir": str(root), "training": {name: value["best_epoch"] for name, value in training.items()}, "adapter_fit_quality": adapter_quality, "ae": aggregate_metrics["ae"], "physics_jepa": aggregate_metrics["physics_jepa"], "paired_comparison": paired_summary, "interpolation": interpolation}, indent=2))


if __name__ == "__main__":
    main()
