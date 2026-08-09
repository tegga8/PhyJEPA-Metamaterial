"""Raw SUTD PRT discovery and reproducible Phase 1 subset construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FAMILIES = ("PLG", "PLR", "PTN", "RDN")


@dataclass(frozen=True)
class Shard:
    family: str
    name: str
    image_path: Path
    curve_path: Path
    length: int


def _numeric_name(path: Path) -> tuple[int, str]:
    digits = "".join(character for character in path.name if character.isdigit())
    return int(digits) if digits else -1, path.name


def _family_root(raw_root: Path, family: str) -> Path:
    candidate = raw_root / f"{family}DATASET"
    options = [candidate, candidate / candidate.name]
    for option in options:
        if (option / "full_data_list").is_dir():
            return option
    raise FileNotFoundError(f"Could not find {family}DATASET/full_data_list beneath {raw_root}")


def discover_shards(raw_root: str | Path, families: Iterable[str] = FAMILIES) -> list[Shard]:
    """Discover and validate paired image/curve NumPy shards."""
    raw_root = Path(raw_root)
    shards: list[Shard] = []
    for family in families:
        full_data = _family_root(raw_root, family) / "full_data_list"
        for shard_dir in sorted((path for path in full_data.iterdir() if path.is_dir()), key=_numeric_name):
            images = sorted(shard_dir.glob("Integrate_image_*.npy"))
            curves = sorted(shard_dir.glob("Integrate_curve_*.npy"))
            if len(images) != 1 or len(curves) != 1:
                raise ValueError(f"Expected one image and curve file in {shard_dir}")
            image = np.load(images[0], mmap_mode="r", allow_pickle=False)
            curve = np.load(curves[0], mmap_mode="r", allow_pickle=False)
            if image.ndim != 2 or image.shape[1] != 256:
                raise ValueError(f"Unexpected geometry shape in {images[0]}: {image.shape}")
            if curve.ndim != 3 or curve.shape[1:] != (2, 1001):
                raise ValueError(f"Unexpected response shape in {curves[0]}: {curve.shape}")
            if len(image) != len(curve):
                raise ValueError(f"Unpaired sample counts in {shard_dir}: {len(image)} vs {len(curve)}")
            shards.append(Shard(family, shard_dir.name, images[0], curves[0], len(image)))
    return shards


def source_id(shard: Shard, offset: int) -> str:
    return f"{shard.family}/{shard.name}/{offset:06d}"


def _balanced_refs(shards: list[Shard], samples_per_family: int, seed: int) -> list[tuple[Shard, int]]:
    generator = np.random.default_rng(seed)
    refs: list[tuple[Shard, int]] = []
    for family_index, family in enumerate(FAMILIES):
        family_shards = [shard for shard in shards if shard.family == family]
        total = sum(shard.length for shard in family_shards)
        if total < samples_per_family:
            raise ValueError(f"{family} provides {total} samples; need {samples_per_family}")
        selected = generator.choice(total, size=samples_per_family, replace=False)
        boundaries = np.cumsum([shard.length for shard in family_shards])
        for flat_index in selected:
            shard_index = int(np.searchsorted(boundaries, flat_index, side="right"))
            start = 0 if shard_index == 0 else int(boundaries[shard_index - 1])
            refs.append((family_shards[shard_index], int(flat_index - start)))
    generator.shuffle(refs)
    return refs


def _split_ids(source_ids: list[str], seed: int) -> dict[str, list[str]]:
    """Split each design family 80/10/10, then combine, preserving balance."""
    generator = np.random.default_rng(seed)
    result = {"train": [], "val": [], "test": []}
    for family in FAMILIES:
        family_ids = [item for item in source_ids if item.startswith(f"{family}/")]
        permutation = generator.permutation(len(family_ids))
        n_train = int(0.8 * len(family_ids))
        n_val = int(0.1 * len(family_ids))
        for split, indexes in {
            "train": permutation[:n_train],
            "val": permutation[n_train : n_train + n_val],
            "test": permutation[n_train + n_val :],
        }.items():
            result[split].extend(family_ids[int(index)] for index in indexes)
    for ids in result.values():
        generator.shuffle(ids)
    return result


def build_subset(
    raw_root: str | Path,
    output_root: str | Path,
    size: int = 5_000,
    seed: int = 42,
    base_subset_root: str | Path | None = None,
) -> dict[str, int]:
    """Materialize a balanced subset without modifying raw data.

    When ``base_subset_root`` is supplied, the new subset is a nested extension:
    every base source ID remains in its existing split.  This supports a fair
    data-scale experiment without changing the completed 5k split.
    """
    if size % len(FAMILIES):
        raise ValueError(f"Subset size must be divisible by {len(FAMILIES)} for family balance")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    shards = discover_shards(raw_root)
    split_override: dict[str, list[str]] | None = None
    if base_subset_root is None:
        refs = _balanced_refs(shards, size // len(FAMILIES), seed)
        selection = "uniform random without replacement within each of PLG, PLR, PTN, RDN"
    else:
        base_root = Path(base_subset_root)
        base_metadata = json.loads((base_root / "metadata.json").read_text(encoding="utf-8"))
        base_ids = (base_root / "source_ids.txt").read_text(encoding="utf-8").splitlines()
        if len(base_ids) >= size:
            raise ValueError("Nested subset size must exceed the base subset size")
        shard_lookup = {(shard.family, shard.name): shard for shard in shards}

        def ref_from_id(identifier: str) -> tuple[Shard, int]:
            family, shard_name, offset_text = identifier.split("/")
            try:
                shard = shard_lookup[(family, shard_name)]
            except KeyError as exc:
                raise ValueError(f"Base subset references unknown raw shard: {identifier}") from exc
            offset = int(offset_text)
            if not 0 <= offset < shard.length:
                raise ValueError(f"Base subset references invalid raw offset: {identifier}")
            return shard, offset

        refs = [ref_from_id(identifier) for identifier in base_ids]
        selected_ids = set(base_ids)
        split_override = {
            split: (base_root / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            for split in ("train", "val", "test")
        }
        per_family = size // len(FAMILIES)
        generator = np.random.default_rng(seed + 1)
        for family in FAMILIES:
            base_family = [identifier for identifier in base_ids if identifier.startswith(f"{family}/")]
            additional = per_family - len(base_family)
            if additional < 0:
                raise ValueError(f"Base subset has too many {family} samples for target size")
            candidates = [
                (shard, offset)
                for shard in shards if shard.family == family
                for offset in range(shard.length)
                if source_id(shard, offset) not in selected_ids
            ]
            if len(candidates) < additional:
                raise ValueError(f"Not enough unused {family} samples for nested subset")
            selected = generator.choice(len(candidates), size=additional, replace=False)
            additional_refs = [candidates[int(index)] for index in selected]
            refs.extend(additional_refs)
            additional_ids = [source_id(shard, offset) for shard, offset in additional_refs]
            desired = {"train": int(0.8 * per_family), "val": int(0.1 * per_family), "test": per_family - int(0.8 * per_family) - int(0.1 * per_family)}
            generator.shuffle(additional_ids)
            cursor = 0
            for split in ("train", "val", "test"):
                base_count = sum(identifier.startswith(f"{family}/") for identifier in split_override[split])
                needed = desired[split] - base_count
                if needed < 0:
                    raise ValueError(f"Base {split} split has too many {family} samples")
                split_override[split].extend(additional_ids[cursor : cursor + needed])
                cursor += needed
            if cursor != len(additional_ids):
                raise RuntimeError("Nested split accounting did not consume all additional IDs")
        for ids in split_override.values():
            generator.shuffle(ids)
        selection = f"nested extension of {base_root.as_posix()} with base split IDs preserved; additional samples uniform without replacement per family"

    geometry_out = np.lib.format.open_memmap(
        output_root / "geometries.npy", mode="w+", dtype=np.uint8, shape=(size, 1, 16, 16)
    )
    response_out = np.lib.format.open_memmap(
        output_root / "responses.npy", mode="w+", dtype=np.float32, shape=(size, 4, 1001)
    )
    opened: dict[Path, np.ndarray] = {}
    source_ids: list[str] = []
    for position, (shard, offset) in enumerate(refs):
        if shard.image_path not in opened:
            opened[shard.image_path] = np.load(shard.image_path, mmap_mode="r")
        if shard.curve_path not in opened:
            opened[shard.curve_path] = np.load(shard.curve_path, mmap_mode="r")
        image = opened[shard.image_path]
        curve = opened[shard.curve_path]
        geometry_out[position, 0] = image[offset].reshape(16, 16)
        # Complex y-cross (T) and x-co (R) reflection coefficients. T/R are
        # upstream coefficient names, not transmission/reflection labels.
        response_out[position] = np.stack(
            (curve[offset, 0].real, curve[offset, 0].imag, curve[offset, 1].real, curve[offset, 1].imag)
        )
        source_ids.append(source_id(shard, offset))
    del geometry_out, response_out
    np.save(output_root / "frequency_ghz.npy", np.linspace(2.0, 12.0, 1001, dtype=np.float32))

    splits = _split_ids(source_ids, seed) if split_override is None else split_override
    split_dir = output_root / "splits"
    split_dir.mkdir()
    for split, ids in splits.items():
        (split_dir / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    source_to_position = {identifier: position for position, identifier in enumerate(source_ids)}
    response = np.load(output_root / "responses.npy", mmap_mode="r")
    train_positions = np.asarray([source_to_position[item] for item in splits["train"]])
    train = np.asarray(response[train_positions], dtype=np.float64)
    # [channel, frequency] statistics broadcast cleanly onto one response.
    mean = train.mean(axis=(0, 2), keepdims=False).astype(np.float32)[:, None]
    std = train.std(axis=(0, 2), keepdims=False).astype(np.float32)[:, None]
    np.savez_compressed(output_root / "train_response_stats.npz", mean=mean, std=np.maximum(std, 1e-8))
    (output_root / "source_ids.txt").write_text("\n".join(source_ids) + "\n", encoding="utf-8")
    metadata = {
        "dataset": "SUTD PRT (repository is named SUTD_PRCM_dataset)",
        "seed": seed,
        "subset_size": size,
        "families": list(FAMILIES),
        "geometry_shape": [1, 16, 16],
        "response_shape": [4, 1001],
        "response_channels": [
            "y_cross_reflection_real",
            "y_cross_reflection_imag",
            "x_co_reflection_real",
            "x_co_reflection_imag",
        ],
        "frequency_axis": {"start_ghz": 2.0, "stop_ghz": 12.0, "points": 1001, "step_ghz": 0.01},
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "selection": selection,
        "base_subset_root": str(base_subset_root) if base_subset_root is not None else None,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {name: len(ids) for name, ids in splits.items()}
