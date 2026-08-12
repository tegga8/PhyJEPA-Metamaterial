"""Assemble the evidence-backed Phase 6.1 report after all validation stages."""
from __future__ import annotations

import csv
import json
from importlib.metadata import version
from pathlib import Path
import platform
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(value: object, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}g}" if isinstance(value, (float, int)) else str(value)


def main() -> None:
    output = ROOT / "outputs" / "phase6_1"
    config = read_json(output / "config.json")
    convergence = read_json(output / "convergence" / "metrics.json")
    calibration = read_json(output / "calibration" / "aggregate_metrics.json")
    comparison = read_json(output / "comparisons" / "aggregate_metrics.json")
    generated = read_json(output / "generated" / "metrics.json")
    sanity = read_json(output / "sanity" / "metrics.json")
    sweep = read_csv(output / "calibration" / "substrate_sweep.csv")
    complexity = read_csv(output / "comparisons" / "complexity_summary.csv")
    pareto = read_csv(output / "generated" / "pareto_summary.csv")
    convergence_established = convergence["status"] == "CONVERGENCE ESTABLISHED"
    physical_mapping_agrees = calibration["physical_and_empirical_mapping_agree"]
    rcwa_dataset = comparison["aggregate"]["rcwa_vs_dataset"]["normalized_mse"]
    cnn_dataset = comparison["aggregate"]["cnn_vs_dataset"]["normalized_mse"]
    cnn_rcwa = comparison["aggregate"]["cnn_vs_rcwa"]["normalized_mse"]
    relationship = generated["target_error_relationship"]
    # A is intentionally never auto-awarded: an agreement-selected substrate
    # remains an independent-model calibration rather than source confirmation.
    classification = "C" if (not convergence_established or not physical_mapping_agrees or relationship["spearman"] is None or relationship["spearman"] < .5) else "B"
    limitation = (
        "Fourier convergence was not established." if not convergence_established else
        "The empirical stored-data mapping differs from the physical p/s mapping." if not physical_mapping_agrees else
        "The substrate thickness is agreement-calibrated rather than established from the original FEM source."
    )
    max_reflected = max(float(item["reflected_power_max"]) for item in sanity["tests"].values())
    max_transmitted = max(float(item["transmitted_power_max"]) for item in sanity["tests"].values())
    metrics = {
        "phase": "6.1",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "meent": version("meent"), "backend": "PyTorch", "device": "cuda" if torch.cuda.is_available() else "cpu", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "frequency": {"range_ghz": [2.0, 12.0], "points": 1001},
        "physical_setup": {"period_mm": [10.0, 10.0], "patch_size_mm": .5, "patch_thickness_mm": .018, "substrate_epsilon_r": 2.65, "substrate_loss_tangent": .003, "substrate_thickness_selected_for_agreement_mm": calibration["thickness_selection"]["selected_for_agreement_mm"], "backing_thickness_mm": .18, "incidence": "normal", "polarization": "x, meent TM/p input"},
        "convergence": convergence,
        "sanity": sanity,
        "dataset_calibration": calibration,
        "comparison": comparison,
        "generated": generated,
        "classification": classification,
        "main_limitation": limitation,
        "recommendation": "Do not use the frozen CNN physics loss for unconstrained inverse design." if classification == "C" else "Improve calibration/uncertainty handling before any large inverse-design campaign.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = f"""# Phase 6.1 — Full-Spectrum RCWA Calibration, Convergence, and Surrogate Validation

## Objective

This phase validates an independent Python RCWA calculation before using the frozen Phase 2.5 CNN as an inverse-design physics objective. Stored dataset responses are original CST FEM outputs; RCWA is an independent numerical calculation, not the original ground truth.

## Why Phase 6 was insufficient

The preceding run used three frequencies, three geometries, incomplete Fourier convergence, and only three candidates per target. It was correctly classified C and did not establish the required implication from lower CNN target loss to lower independently evaluated target loss.

## Environment

- Python `{platform.python_version()}`, PyTorch `{torch.__version__}`, meent `{version('meent')}`, PyTorch backend.
- Execution device available: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`. The solver records the actual requested device and cache metadata per configuration.
- Full spectrum: 2.00–12.00 GHz, 1001 actual RCWA frequency points, 0.01 GHz spacing.

## Physical configuration and geometry mapping

The 16×16 binary stored geometry maps to a 20×20 raster of 0.5 mm cells with a two-cell (1 mm) air border in a 10×10 mm period. The patch/back copper thicknesses are 0.018/0.18 mm. The independent model uses εr=2.65 with loss tangent 0.003, μr=1, and finite-conductivity copper (σ=5.8e7 S/m). The selected substrate thickness ({value(calibration['thickness_selection']['selected_for_agreement_mm'])} mm) is selected only for stored-data agreement; it is not claimed as the original CST parameter.

## Polarization convention

Physical mapping: `{calibration['physical_mapping']}`. At normal incidence, the implementation excites meent’s TM/p input for x polarization and reads zero-order p as co-reflected x (`Rx`) and zero-order s as cross-reflected y (`Ty`). The empirical mapping giving the lowest stored-data comparison MSE at the selected thickness is `{calibration['empirical_best_mapping_for_comparison']['channel_mapping']}`. Agreement: `{physical_mapping_agrees}`. Both mappings are retained in `calibration/polarization_mapping_diagnostics.csv`; no mapping was silently substituted for the physical one.

## Fourier convergence

Criterion: `{convergence['criterion']['name']}`. Status: **{convergence['status']}**. Selected production order: `{convergence['selected_order']}`. The full per-order, per-channel metrics and representative plots are in `outputs/phase6_1/convergence/`.

## Sanity tests and power diagnostic

Empty cell, uniform metal, and symmetric-square cases were run across the full spectrum. Max reflected power: {max_reflected:.6f}; max transmitted power: {max_transmitted:.6g}. {sanity['power_definition']}

## Nine-geometry full-spectrum calibration

The deterministic seed-42 manifest contains three simple, three medium, and three complex source-ID-disjoint held-out test geometries. The physical-map sweep is summarized below.

| Thickness (mm) | Mean normalized MSE | Median normalized MSE |
|---:|---:|---:|
{chr(10).join(f"| {row['substrate_thickness_mm']} | {float(row['mean_normalized_mse']):.6g} | {float(row['median_normalized_mse']):.6g} |" for row in sweep)}

## RCWA, CNN, and stored dataset

| Comparison | Overall normalized MSE |
|---|---:|
| RCWA → stored dataset | {rcwa_dataset:.6g} |
| CNN → stored dataset | {cnn_dataset:.6g} |
| CNN → RCWA | {cnn_rcwa:.6g} |

Per-channel, magnitude, frequency-wise, resonance, and three representative full-spectrum plots are saved under `outputs/phase6_1/comparisons/`.

## Generated geometry validation

Actual evaluated counts: Phase 5A {generated['counts']['phase5a']}; Phase 5B small {generated['counts']['phase5b_small']}; Phase 5B medium {generated['counts']['phase5b_medium']}. Each candidate used all 1001 RCWA frequencies.

Pooled target-error relationship: Pearson {value(relationship['pearson'])}, Spearman {value(relationship['spearman'])}, Kendall τ {value(relationship['kendall_tau'])}, pairwise ordering agreement {value(relationship['pairwise_ordering_agreement'])}, top-1 {value(relationship['top_1_overlap'])}, top-3 {value(relationship['top_3_overlap'])}, top-5 {value(relationship['top_5_overlap'])}. The per-target rankings have only three model candidates, so unavailable top-k values are deliberately not inferred.

The largest `RCWA target MSE − CNN target MSE` exploitation gap is {value(generated['worst_exploitation_gap'])}; saved worst cases include geometry, target, CNN, and RCWA spectra.

## Complexity and Phase 5B physical Pareto subset

| Group | Samples | CNN–RCWA MSE | RCWA target MSE |
|---|---:|---:|---:|
{chr(10).join(f"| {row['complexity_group']} | {row['sample_count']} | {value(float(row['mean_cnn_vs_rcwa_normalized_mse']))} | {value(float(row['mean_rcwa_vs_dataset_normalized_mse']))} |" for row in complexity)}

The independently evaluated Pareto subset is saved in `outputs/phase6_1/generated/pareto_summary.csv`; it keeps the existing surrogate Pareto value separate from RCWA target error.

## Limitations, classification, and recommendation

**Final classification: {classification}.** Main limitation: {limitation}

Recommendation: {metrics['recommendation']}

Stop after this validation phase. Do not begin unconstrained inverse design, retrain a model, or modify the Phase 5 architectures based on this report alone.
"""
    destination = ROOT / "docs" / "phase6_1_full_spectrum_rcwa_validation.md"
    destination.write_text(report, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
