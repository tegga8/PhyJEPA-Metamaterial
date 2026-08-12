"""Evaluate a trained Physics-JEPA representation (representation gate).

Freezes the trained encoders and runs the full fixed battery:
weak response/resonance probes, physics-similarity distance correlations,
within/cross-family correlation, correct-vs-shuffled control (when a shuffled
model directory is provided), latent-size comparison (when the 64-D model
directory is provided), and representative pair mining.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics_jepa_metrics_eval import evaluate_model_representation, load_cached_latents
from src.physics_jepa_plots import (
    plot_correct_vs_shuffled,
    plot_distance_correlation,
    plot_latent_size_comparison,
    plot_loss_curve,
    plot_probe_response,
    plot_representative_pairs,
    plot_resonance_probe,
    plot_variance_diagnostics,
    plot_within_cross_family,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path, default=None, help="shuffled-pair control model directory")
    parser.add_argument("--size-compare-dir", type=Path, default=None, help="other latent-size model directory")
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_30k"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-pairs", type=int, default=20000)
    parser.add_argument("--num-pairs-family", type=int, default=40000)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = args.output_dir or args.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = evaluate_model_representation(
        model_dir=args.model_dir,
        subset_root=args.subset_root,
        device=args.device,
        num_pairs=args.num_pairs,
        num_pairs_family=args.num_pairs_family,
        probe_epochs=args.probe_epochs,
        seed=args.seed,
    )

    if args.compare_dir is not None:
        comparison: dict[str, float] = {}
        for name, directory in (("correct", args.model_dir), ("shuffled", args.compare_dir)):
            cached = load_cached_latents(directory, "test")
            metrics = evaluate_model_representation(
                model_dir=directory,
                subset_root=args.subset_root,
                device=args.device,
                num_pairs=args.num_pairs,
                num_pairs_family=args.num_pairs_family,
                probe_epochs=args.probe_epochs,
                seed=args.seed,
                load_only_similarity=cached is not None,
            )
            comparison[f"{name}_z_pred_latent_vs_response_spearman"] = metrics["similarity"]["z_pred"]["latent_vs_response_spearman"]
        comparison["separation"] = float(comparison["correct_z_pred_latent_vs_response_spearman"] - comparison["shuffled_z_pred_latent_vs_response_spearman"])
        evaluation["correct_vs_shuffled"] = comparison
        plot_correct_vs_shuffled(comparison, output_dir / "plots" / "correct_vs_shuffled.png")

    if args.size_compare_dir is not None:
        summaries = {
            "32": {key: evaluation["similarity"]["z_pred"][key] for key in ("latent_vs_response_spearman", "latent_vs_geometry_spearman")} | {"response_r2_vs_mean": evaluation["probes"]["z_target"]["response"]["response_r2_vs_mean"]},
        }
        other = evaluate_model_representation(
            model_dir=args.size_compare_dir,
            subset_root=args.subset_root,
            device=args.device,
            num_pairs=args.num_pairs,
            num_pairs_family=args.num_pairs_family,
            probe_epochs=args.probe_epochs,
            seed=args.seed,
        )
        summaries[str(other["latent_dim"])] = {key: other["similarity"]["z_pred"][key] for key in ("latent_vs_response_spearman", "latent_vs_geometry_spearman")} | {"response_r2_vs_mean": other["probes"]["z_target"]["response"]["response_r2_vs_mean"]}
        evaluation["latent_size_comparison"] = summaries
        plot_latent_size_comparison(summaries, output_dir / "plots" / "latent_size_comparison.png")

    history = args.model_dir / "training_history.csv"
    if history.is_file():
        plot_loss_curve(history, output_dir / "plots" / "jepa_loss_curve.png")
        plot_variance_diagnostics(history, output_dir / "plots" / "latent_variance_diagnostics.png")

    for latent_name in ("z_target", "z_pred", "z_geometry"):
        probe = evaluation["probes"][latent_name]
        if probe["response"]["has_data"]:
            cached = load_cached_latents(args.model_dir, "test")
            plot_probe_response(
                probe["response"]["predictions"],
                cached["response"],
                probe["response"]["baseline_mean"],
                sample_index=0,
                output=output_dir / "plots" / f"response_probe_{latent_name}.png",
            )
        plot_resonance_probe(probe["resonance"], output_dir / "plots" / f"resonance_probe_{latent_name}.png")

    similarity = evaluation["similarity"]
    cached = load_cached_latents(args.model_dir, "test")
    for latent_name in ("z_pred", "z_target"):
        distances = evaluation["pair_distances"][latent_name]
        plot_distance_correlation(
            distances["d_latent"],
            distances["d_response"],
            distances["d_geometry"],
            similarity[latent_name]["latent_vs_response_spearman"],
            similarity[latent_name]["latent_vs_geometry_spearman"],
            output_dir / "plots" / f"distance_correlation_{latent_name}.png",
        )
    if "within_cross_family" in evaluation:
        plot_within_cross_family(evaluation["within_cross_family"], output_dir / "plots" / "within_cross_family.png")
    plot_representative_pairs(
        cached["geometry"],
        cached["response"],
        evaluation["representative_pairs"]["case_a"],
        output_dir / "plots" / "case_a_similar_response_different_geometry.png",
        output_dir / "plots" / "case_b_similar_geometry_different_response.png",
    )

    evaluation["python"] = platform.python_version()
    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(jsonable(evaluation), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(jsonable(evaluation), indent=2))
    print(json.dumps({"report": str(report_path), "output_dir": str(output_dir)}))


def jsonable(value: object) -> object:
    """Convert numpy values for JSON while summarizing large prediction arrays."""
    if isinstance(value, np.ndarray):
        if value.size <= 5000:
            return value.tolist()
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
