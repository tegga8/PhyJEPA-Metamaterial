"""Stage A of Phase 6.1: full-spectrum, three-geometry Fourier convergence."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.rcwa_solver import RCWAConfig, frequency_vector
from src.rcwa_validation import (
    CHANNELS,
    PHYSICAL_MAPPING,
    cached_solve,
    convergence_row,
    packed,
    save_json,
    select_phase42_balanced_indices,
)


# A conservative absolute MSE threshold for adjacent packed complex spectra.
# Convergence is established only when the N=9 -> N=11 transition meets this
# threshold for every simple/medium/complex representative.
SUCCESSIVE_ORDER_MSE_THRESHOLD = 1e-4


def write_manifest(dataset: SUTDPRCMDataset, selected: np.ndarray, labels: list[str], scores: np.ndarray, subset_root: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for local_index in selected:
        global_index = int(dataset.indices[int(local_index)])
        rows.append({
            "source_id": dataset.source_id(int(local_index)),
            "test_index": int(local_index),
            "global_index": global_index,
            "complexity_group": labels[int(local_index)],
            "complexity_score": float(scores[int(local_index)]),
            "geometry_path": f"{subset_root.as_posix()}/geometries.npy[{global_index}]",
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_1"))
    parser.add_argument("--orders", default="1,3,5,7,9,11")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu", help="CPU multiprocessing is benchmarked separately because meent cannot batch wavelengths.")
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-only", action="store_true", help="Write the deterministic nine-geometry calibration manifest without running RCWA.")
    args = parser.parse_args()

    orders = [int(value) for value in args.orders.split(",")]
    if orders != sorted(orders) or len(orders) < 2 or any(order < 0 for order in orders):
        raise ValueError("--orders must contain at least two non-negative, increasing orders")
    frequencies = frequency_vector()
    dataset = SUTDPRCMDataset(args.subset_root, "test", normalize_response=False)
    all_geometries = np.asarray([dataset.geometries[int(index)] for index in dataset.indices])
    # The full calibration manifest is nine source-ID-disjoint held-out
    # geometries (3 per Phase-4.2 complexity group).  Stage A itself uses a
    # deterministic one-per-group subset, so convergence remains tractable
    # while all later stages have the required nine-geometry manifest.
    manifest_indices, manifest_labels, manifest_scores = select_phase42_balanced_indices(
        all_geometries, per_complexity=3, seed=args.seed
    )
    write_manifest(dataset, manifest_indices, manifest_labels, manifest_scores, args.subset_root, args.output_dir / "validation_geometry_manifest.csv")
    if args.manifest_only:
        print(args.output_dir / "validation_geometry_manifest.csv")
        return
    selected, labels, scores = select_phase42_balanced_indices(all_geometries, per_complexity=1, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out = args.output_dir / "convergence"
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    final_errors: list[float] = []
    for local_index in selected:
        geometry, _ = dataset[int(local_index)]
        previous: np.ndarray | None = None
        per_geometry: list[dict[str, object]] = []
        for order in orders:
            config = RCWAConfig(fourier_order=order, device=args.device, cpu_workers=args.cpu_workers)
            result = cached_solve(geometry.numpy(), frequencies, config, args.output_dir / "cache")
            current = packed(result)
            detail = convergence_row(current, previous)
            row: dict[str, object] = {
                "source_id": dataset.source_id(int(local_index)),
                "test_index": int(local_index),
                "complexity_group": labels[int(local_index)],
                "complexity_score": float(scores[int(local_index)]),
                "fourier_order": order,
                "frequency_points": len(frequencies),
                "runtime_seconds": float(result.metadata["runtime_seconds"]),
                "cache_hit": bool(result.metadata.get("cache_hit", False)),
                "physical_channel_mapping": PHYSICAL_MAPPING,
                **detail,
            }
            rows.append(row)
            per_geometry.append(row)
            np.savez_compressed(
                out / f"{labels[int(local_index)]}_order_{order}.npz",
                frequency_ghz=frequencies,
                packed_response=current,
                reflected_power=result.reflected_power,
                transmitted_power=result.transmitted_power,
            )
            previous = current
        final_errors.append(float(per_geometry[-1]["complex_response_mse"]))
        figure, axis = plt.subplots(figsize=(6, 3.5))
        axis.semilogy(orders[1:], [float(item["complex_response_mse"]) for item in per_geometry[1:]], marker="o", label="packed")
        for channel in CHANNELS:
            axis.semilogy(orders[1:], [float(item[f"{channel}_mse"]) for item in per_geometry[1:]], marker=".", label=channel)
        axis.axhline(SUCCESSIVE_ORDER_MSE_THRESHOLD, color="black", linestyle="--", linewidth=.8, label="criterion")
        axis.set(xlabel="Fourier order N", ylabel="successive full-spectrum MSE", title=f"{labels[int(local_index)]}: {dataset.source_id(int(local_index))}")
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(plots / f"{labels[int(local_index)]}_convergence.png", dpi=180)
        plt.close(figure)

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    converged = bool(all(error <= SUCCESSIVE_ORDER_MSE_THRESHOLD for error in final_errors))
    summary = {
        "frequency_ghz": frequencies,
        "frequency_points": len(frequencies),
        "orders": orders,
        "criterion": {
            "name": "all three N=9-to-N=11 packed full-spectrum MSE <= 1e-4",
            "threshold": SUCCESSIVE_ORDER_MSE_THRESHOLD,
        },
        "status": "CONVERGENCE ESTABLISHED" if converged else "NO CONVERGENCE ESTABLISHED",
        "selected_order": orders[-1] if converged else None,
        "physical_channel_mapping": PHYSICAL_MAPPING,
        "rows": rows,
    }
    save_json(out / "metrics.json", summary)
    save_json(args.output_dir / "config.json", {
        "phase": "6.1",
        "seed": args.seed,
        "frequency_ghz": frequencies,
        "frequency_points": len(frequencies),
        "device_request": args.device,
        "cpu_workers": args.cpu_workers,
        "convergence": {key: summary[key] for key in ("orders", "criterion", "status", "selected_order")},
        "physical_channel_mapping": PHYSICAL_MAPPING,
    })
    print(out / "metrics.json")


if __name__ == "__main__":
    main()
