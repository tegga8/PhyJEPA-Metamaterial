"""Run the Phase 6 RCWA sanity checks with physical power diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rcwa_solver import RCWAConfig, frequency_vector, solve_geometry, solve_physical_pattern
from src.rcwa_validation import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_1"))
    parser.add_argument("--fourier-order", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true", help="Use 2, 7, 12 GHz only for installation checks; default is all 1001 points.")
    args = parser.parse_args()
    frequencies = np.array([2.0, 7.0, 12.0]) if args.quick else frequency_vector()
    config = RCWAConfig(fourier_order=args.fourier_order, device=args.device)
    sanity_dir = args.output_dir / "sanity"; sanity_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "empty_pattern": solve_geometry(np.zeros((16, 16)), frequencies, config=config),
        "uniform_metal_unit_cell": solve_physical_pattern(np.ones((20, 20), dtype=bool), frequencies, config=config),
        "symmetric_square": solve_geometry(np.pad(np.ones((8, 8)), 4), frequencies, config=config),
    }
    report = {"frequency_points": int(len(frequencies)), "fourier_order": args.fourier_order, "device_requested": args.device, "tests": {}}
    for name, result in cases.items():
        np.savez_compressed(sanity_dir / f"{name}.npz", ty=result.ty, rx=result.rx, reflected_power=result.reflected_power, transmitted_power=result.transmitted_power)
        power = result.reflected_power + result.transmitted_power
        report["tests"][name] = {
            "max_abs_ty": float(np.abs(result.ty).max()),
            "reflected_power_min": float(result.reflected_power.min()),
            "transmitted_power_max": float(result.transmitted_power.max()),
            "power_max": float(power.max()),
            "power_min": float(power.min()),
            "reflected_power_max": float(result.reflected_power.max()),
            "passivity_within_1_percent": bool(np.all(power <= 1.01)),
            "metadata": result.metadata,
        }
    report["power_definition"] = "meent de_ri/de_ti are summed diffraction efficiencies; the reported total is reflected_power + transmitted_power. It is a diagnostic, not an imposed |R|^2+|T|^2 identity."
    save_json(sanity_dir / "metrics.json", report)
    print(sanity_dir / "metrics.json")


if __name__ == "__main__":
    main()
