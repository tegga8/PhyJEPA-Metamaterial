"""Independent Python RCWA evaluation for the SUTD reflective metasurface.

This module deliberately keeps physical complex coefficients separate from the
Phase-2 response normalization.  It uses meent's PyTorch backend when CUDA is
available and otherwise falls back to CPU.  The returned ``Ty`` is the
y-polarized *cross-reflected* coefficient, not ordinary transmission.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable

import numpy as np


C_MM_GHZ = 299.792458
EPSILON_0_F_PER_M = 8.8541878128e-12


@dataclass(frozen=True)
class RCWAConfig:
    """Physical and numerical configuration for an independent RCWA run.

    The substrate thickness is intentionally a calibration parameter; it is
    not claimed to be the original CST substrate thickness.  Meent uses the
    exp(+i omega t) material convention, so the source-style ``+i`` loss
    notation is converted to a negative imaginary permittivity internally.
    """

    period_mm: tuple[float, float] = (10.0, 10.0)
    patch_size_mm: float = 0.5
    padding_mm: float = 1.0
    patch_thickness_mm: float = 0.018
    backing_thickness_mm: float = 0.18
    substrate_epsilon_r: float = 2.65
    substrate_loss_tangent: float = 0.003
    substrate_thickness_mm: float = 0.20
    copper_conductivity_s_per_m: float = 5.8e7
    copper_relative_permeability: float = 1.0
    fourier_order: int = 3
    device: str = "auto"
    complex_precision: str = "auto"
    use_pinv: bool = True
    cpu_workers: int = 1


@dataclass
class RCWASolveResult:
    """Physical complex response and power diagnostics from one geometry."""

    ty: np.ndarray
    rx: np.ndarray
    reflected_power: np.ndarray
    transmitted_power: np.ndarray
    metadata: dict[str, Any]

    def packed_response(self) -> np.ndarray:
        return pack_response(self.ty, self.rx)


def frequency_vector() -> np.ndarray:
    """Return the authoritative processed-project frequency vector in GHz."""
    return np.linspace(2.0, 12.0, 1001, dtype=np.float64)


def validate_frequency_vector(frequencies_ghz: Iterable[float] | np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies_ghz, dtype=np.float64)
    if frequencies.ndim != 1 or not len(frequencies):
        raise ValueError("frequencies_ghz must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("frequencies_ghz must contain finite positive values")
    return frequencies


def pack_response(ty: np.ndarray, rx: np.ndarray) -> np.ndarray:
    """Pack physical complex coefficients as [Re(Ty), Im(Ty), Re(Rx), Im(Rx)]."""
    ty_array, rx_array = np.asarray(ty), np.asarray(rx)
    if ty_array.ndim != 1 or rx_array.ndim != 1 or ty_array.shape != rx_array.shape:
        raise ValueError("Ty and Rx must be matching one-dimensional complex arrays")
    if not np.iscomplexobj(ty_array) or not np.iscomplexobj(rx_array):
        raise ValueError("Ty and Rx must be complex arrays")
    return np.stack((ty_array.real, ty_array.imag, rx_array.real, rx_array.imag)).astype(np.float32)


def geometry_to_physical_pattern(geometry: np.ndarray) -> np.ndarray:
    """Map a stored 16x16 binary geometry to a padded 10x10 mm 20x20 raster.

    Rows map to y and columns map to x.  A raster element is 0.5 mm, so the
    1 mm border becomes two air cells on each side.
    """
    array = np.asarray(geometry)
    if array.shape == (1, 16, 16):
        array = array[0]
    if array.shape != (16, 16):
        raise ValueError(f"Expected geometry [16,16] or [1,16,16], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Geometry contains non-finite values")
    pattern = np.zeros((20, 20), dtype=bool)
    pattern[2:18, 2:18] = array > 0.5
    return pattern


def copper_relative_permittivity(frequency_ghz: float, conductivity_s_per_m: float = 5.8e7) -> complex:
    """Finite-conductivity copper in meent's passive exp(+i omega t) convention."""
    omega = 2.0 * np.pi * float(frequency_ghz) * 1e9
    return complex(1.0, -conductivity_s_per_m / (EPSILON_0_F_PER_M * omega))


def _refractive_index(epsilon_r: complex) -> complex:
    value = np.sqrt(np.complex128(epsilon_r))
    # Keep the positive-real branch.  For meent's exp(+i omega t) convention
    # passive media have Im(n) <= 0; forcing Im(n) >= 0 would turn loss into
    # gain and can produce reflected power above one.
    return value if value.real >= 0 else -value


def substrate_refractive_index(config: RCWAConfig) -> complex:
    return _refractive_index(config.substrate_epsilon_r * (1.0 + 1j * config.substrate_loss_tangent))


def copper_refractive_index(frequency_ghz: float, config: RCWAConfig) -> complex:
    # mu_r is retained explicitly in the metadata/model even though it is one.
    return _refractive_index(copper_relative_permittivity(frequency_ghz, config.copper_conductivity_s_per_m) * config.copper_relative_permeability)


def resolve_device(requested: str = "auto") -> tuple[str, str]:
    """Select CUDA only when Torch can actually execute CUDA kernels."""
    import torch

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
    label = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    return device, label


def _torch_complex_precision(config: RCWAConfig, device: str):
    import torch

    if config.complex_precision == "complex64":
        return torch.complex64, "complex64"
    if config.complex_precision in {"auto", "complex128"}:
        return torch.complex128, "complex128"
    raise ValueError("complex_precision must be 'auto', 'complex64', or 'complex128'")


def _unit_cell_refractive_index(pattern: np.ndarray, frequency_ghz: float, config: RCWAConfig) -> np.ndarray:
    copper_n = copper_refractive_index(frequency_ghz, config)
    substrate_n = substrate_refractive_index(config)
    patch = np.where(pattern, copper_n, 1.0 + 0.0j)
    substrate = np.full(pattern.shape, substrate_n, dtype=np.complex128)
    backing = np.full(pattern.shape, copper_n, dtype=np.complex128)
    # Layer order is top -> bottom.  meent propagates them bottom -> top.
    return np.stack((patch, substrate, backing))


def _solve_cpu_frequency_chunk(pattern: np.ndarray, frequencies: np.ndarray, config_data: dict[str, Any]) -> RCWASolveResult:
    """Worker entry point for independent CPU frequencies (Windows-safe/picklable)."""
    # Each worker uses one Torch thread so the requested process count, rather
    # than nested BLAS threads, controls CPU parallelism.
    import torch
    torch.set_num_threads(1)
    config_data["cpu_workers"] = 1
    return solve_physical_pattern(pattern, frequencies, config=RCWAConfig(**config_data))


def _patch_meent_cuda_incidence(solver: Any) -> None:
    """Fix meent 0.12's NumPy-on-CUDA incidence helper at normal incidence.

    The release calls ``np.sin`` on a CUDA tensor in ``get_kx_ky_vector``.
    This local adapter is mathematically identical but stays in Torch, avoiding
    a host transfer and allowing the installed PyTorch backend to use CUDA.
    """
    import torch

    def get_kx_ky_vector(instance: Any, wavelength: Any):
        fto_x = torch.arange(-instance.fto[0], instance.fto[0] + 1, device=instance.device, dtype=instance.type_float)
        fto_y = torch.arange(-instance.fto[1], instance.fto[1] + 1, device=instance.device, dtype=instance.type_float)
        phi = instance.phi if instance.phi is not None else torch.tensor(0.0, device=instance.device, dtype=instance.type_complex)
        sin_theta = torch.sin(instance.theta)
        kx = (instance.n_top * sin_theta * torch.cos(phi) + fto_x * (wavelength / instance.period[0])).type(instance.type_complex).conj()
        ky = (instance.n_top * sin_theta * torch.sin(phi) + fto_y * (wavelength / instance.period[1])).type(instance.type_complex).conj()
        return kx, ky

    solver.get_kx_ky_vector = types.MethodType(get_kx_ky_vector, solver)


def solve_physical_pattern(
    pattern: np.ndarray,
    frequencies_ghz: Iterable[float] | np.ndarray | None = None,
    substrate_thickness_mm: float | None = None,
    fourier_order: int | None = None,
    *,
    config: RCWAConfig | None = None,
) -> RCWASolveResult:
    """Solve a padded 20x20 physical pattern and return complex reflection.

    ``Rx`` is the zero-order p/TM reflection coefficient for x-polarized normal
    incidence; ``Ty`` is its zero-order s/TE cross-reflection coefficient.
    Ordinary transmission is returned only as a power diagnostic because the
    copper-backed stack should strongly suppress it.
    """
    try:
        import meent
    except ImportError as exc:  # pragma: no cover - exercised in dependency-only environments
        raise RuntimeError("meent is required; install it with `python -m pip install meent`") from exc

    base = config or RCWAConfig()
    if substrate_thickness_mm is not None or fourier_order is not None:
        base = RCWAConfig(**{**asdict(base), **({"substrate_thickness_mm": float(substrate_thickness_mm)} if substrate_thickness_mm is not None else {}), **({"fourier_order": int(fourier_order)} if fourier_order is not None else {})})
    if base.substrate_thickness_mm <= 0 or base.fourier_order < 0:
        raise ValueError("substrate_thickness_mm must be positive and fourier_order non-negative")
    if base.cpu_workers < 1:
        raise ValueError("cpu_workers must be at least one")

    frequencies = validate_frequency_vector(frequency_vector() if frequencies_ghz is None else frequencies_ghz)
    pattern = np.asarray(pattern, dtype=bool)
    if pattern.shape != (20, 20):
        raise ValueError(f"Expected padded physical pattern [20,20], got {pattern.shape}")
    device, device_label = resolve_device(base.device)
    if device == "cpu" and base.cpu_workers > 1 and len(frequencies) > 1:
        started = time.perf_counter()
        chunks = [chunk for chunk in np.array_split(frequencies, min(base.cpu_workers, len(frequencies))) if len(chunk)]
        config_data = asdict(base)
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            results = list(executor.map(_solve_cpu_frequency_chunk, [pattern] * len(chunks), chunks, [config_data] * len(chunks)))
        metadata = dict(results[0].metadata)
        metadata.update({
            "frequency_ghz": frequencies.tolist(),
            "runtime_seconds": time.perf_counter() - started,
            "cpu_workers": base.cpu_workers,
            "parallel_frequency_chunks": len(chunks),
        })
        return RCWASolveResult(
            ty=np.concatenate([result.ty for result in results]),
            rx=np.concatenate([result.rx for result in results]),
            reflected_power=np.concatenate([result.reflected_power for result in results]),
            transmitted_power=np.concatenate([result.transmitted_power for result in results]),
            metadata=metadata,
        )
    complex_type, precision_label = _torch_complex_precision(base, device)
    ty = np.empty(len(frequencies), dtype=np.complex128)
    rx = np.empty(len(frequencies), dtype=np.complex128)
    reflected_power = np.empty(len(frequencies), dtype=np.float64)
    transmitted_power = np.empty(len(frequencies), dtype=np.float64)
    # Reuse the meent object across the frequency sweep.  The wavelength and
    # dispersive copper raster are explicitly refreshed on every iteration;
    # this is an exact sweep, not a three-point interpolation shortcut.
    first_frequency = float(frequencies[0])
    solver = meent.call_mee(
            backend=2,
            # meent's MeeTorch property setter accepts 0/1 even though its
            # constructor documentation also mentions CPU/CUDA strings.
            device=1 if device == "cuda" else 0,
            type_complex=complex_type,
            n_top=1.0,
            n_bot=1.0,
            theta=0.0,
            phi=0.0,
            psi=0.0,  # meent's TM/p incident basis, aligned with x at normal incidence.
            period=base.period_mm,
            wavelength=C_MM_GHZ / first_frequency,
            ucell=_unit_cell_refractive_index(pattern, first_frequency, base),
            thickness=(base.patch_thickness_mm, base.substrate_thickness_mm, base.backing_thickness_mm),
            fto=(base.fourier_order, base.fourier_order),
            fourier_type=0,
            enhanced_dfs=True,
            # Homogeneous sanity cells can make redundant Fourier subspaces
            # singular; meent's pseudoinverse path handles those exactly.
            use_pinv=base.use_pinv,
        )
    if device == "cuda":
        _patch_meent_cuda_incidence(solver)
        import torch
        torch.cuda.synchronize()
    started = time.perf_counter()

    for index, frequency in enumerate(frequencies):
        solver.wavelength = C_MM_GHZ / float(frequency)
        solver.ucell = _unit_cell_refractive_index(pattern, float(frequency), base)
        result = solver.conv_solve().res_tm_inc
        center = base.fourier_order
        ty[index] = complex(result.R_s[center, center].detach().cpu().item())
        rx[index] = complex(result.R_p[center, center].detach().cpu().item())
        reflected_power[index] = float(result.de_ri.sum().detach().cpu().item())
        transmitted_power[index] = float(result.de_ti.sum().detach().cpu().item())

    if device == "cuda":
        torch.cuda.synchronize()

    metadata: dict[str, Any] = {
        "solver": "meent",
        "backend": "torch",
        "device": device,
        "device_name": device_label,
        "complex_precision": precision_label,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "frequency_ghz": frequencies.tolist(),
        "fourier_order": base.fourier_order,
        "physical_setup": asdict(base),
        "geometry_mapping": "16x16 binary pixels -> 20x20 0.5-mm raster with a two-pixel (1-mm) air border",
        "polarization_mapping": "x-polarized normal incidence uses meent TM/p input; Ty=zero-order R_s (y cross-reflection), Rx=zero-order R_p (x co-reflection)",
        "copper_model": "finite conductivity: epsilon_r(f)=1-i*sigma/(epsilon0*2*pi*f) for meent exp(+i*omega*t) passive convention; sigma=5.8e7 S/m; independent approximation, not a claimed CST material match",
        "loss_convention": "Source notation uses +i loss; meent's passive exp(+i*omega*t) convention is represented with negative imaginary epsilon and refractive index.",
        "runtime_seconds": time.perf_counter() - started,
    }
    return RCWASolveResult(ty=ty, rx=rx, reflected_power=reflected_power, transmitted_power=transmitted_power, metadata=metadata)


def solve_geometry(
    geometry: np.ndarray,
    frequencies_ghz: Iterable[float] | np.ndarray | None = None,
    substrate_thickness_mm: float | None = None,
    fourier_order: int | None = None,
    *,
    config: RCWAConfig | None = None,
) -> RCWASolveResult:
    """Solve one stored 16x16 geometry through the documented physical mapping."""
    return solve_physical_pattern(
        geometry_to_physical_pattern(geometry),
        frequencies_ghz,
        substrate_thickness_mm,
        fourier_order,
        config=config,
    )


def convergence_mse(current: np.ndarray, previous: np.ndarray) -> float:
    """Complex-response MSE used to compare successive Fourier orders."""
    current_array, previous_array = np.asarray(current), np.asarray(previous)
    if current_array.shape != previous_array.shape:
        raise ValueError("Convergence arrays must have identical shapes")
    return float(np.mean(np.abs(current_array - previous_array) ** 2))
