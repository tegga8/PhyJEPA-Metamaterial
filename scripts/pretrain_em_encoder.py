"""Train the Phase 5A EM embedding readout sanity check on paired responses."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.physics_conditioned_spatial_jepa import EMResponseDecoder, EMResponseEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@torch.inference_mode()
def evaluate(
    encoder: EMResponseEncoder,
    decoder: EMResponseDecoder,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    element_squared = 0.0
    zero_squared = 0.0
    elements = 0
    component_abs = torch.zeros(4, device=device)
    component_items = 0
    for _, response in loader:
        response = response.to(device)
        prediction = decoder(encoder(response))
        element_squared += float(torch.sum(torch.square(prediction - response)).item())
        zero_squared += float(torch.sum(torch.square(response)).item())
        elements += response.numel()
        raw_prediction = prediction * std + mean
        raw_target = response * std + mean
        component_abs += torch.abs(raw_prediction - raw_target).sum(dim=(0, 2))
        component_items += response.shape[0] * response.shape[2]
    return {
        "normalized_mse": element_squared / elements,
        "zero_embedding_normalized_mse": zero_squared / elements,
        "reconstruction_beats_zero": bool(element_squared < zero_squared),
        "re_t_y_mae": float(component_abs[0].item() / component_items),
        "im_t_y_mae": float(component_abs[1].item() / component_items),
        "re_r_x_mae": float(component_abs[2].item() / component_items),
        "im_r_x_mae": float(component_abs[3].item() / component_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase5a/em_embedding"))
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.embedding_dim != 128:
        raise ValueError("Phase 5A specifies a 128-dimensional EM embedding")
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: SUTDPRCMDataset(args.subset_root, split, normalize_response=True) for split in ("train", "val", "test")}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", num_workers=0) for split, dataset in datasets.items()}
    stats = np.load(args.subset_root / "train_response_stats.npz")
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)
    encoder = EMResponseEncoder(args.embedding_dim).to(device)
    decoder = EMResponseDecoder(args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW((*encoder.parameters(), *decoder.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path = args.output_dir / "best.pt"
    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        squared = 0.0
        elements = 0
        for _, response in loaders["train"]:
            response = response.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = decoder(encoder(response))
            loss = torch.mean(torch.square(prediction - response))
            loss.backward()
            optimizer.step()
            squared += float(torch.sum(torch.square(prediction.detach() - response)).item())
            elements += response.numel()
        validation = evaluate(encoder, decoder, loaders["val"], device, mean, std)
        record = {"epoch": epoch, "train_normalized_mse": squared / elements, "val_normalized_mse": validation["normalized_mse"]}
        history.append(record)
        print(f"epoch {epoch:03d} | train_mse={record['train_normalized_mse']:.6f} | val_mse={record['val_normalized_mse']:.6f}")
        if validation["normalized_mse"] < best_validation:
            best_validation = validation["normalized_mse"]
            stale_epochs = 0
            torch.save({"encoder_state_dict": encoder.state_dict(), "decoder_state_dict": decoder.state_dict(), "epoch": epoch, "validation": validation, "args": vars(args)}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    elapsed = time.perf_counter() - start
    metrics = evaluate(encoder, decoder, loaders["test"], device, mean, std)
    metrics.update({"best_epoch": int(checkpoint["epoch"]), "best_validation_normalized_mse": float(checkpoint["validation"]["normalized_mse"]), "test_samples": len(datasets["test"])})
    config = {
        "phase": "5A_em_embedding_sanity", "subset_root": str(args.subset_root), "normalization_stats": str(args.subset_root / "train_response_stats.npz"),
        "response_representation": "normalized [Re(T_y), Im(T_y), Re(R_x), Im(R_x)] [4,1001]", "frequency_ghz": [2.0, 12.0, 1001],
        "encoder": "Conv1d 4->32->64->128 with GroupNorm/GELU/global-average-pool; Linear 128->128", "decoder": "Linear 128->4004", "embedding_dim": args.embedding_dim,
        "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "epochs_requested": args.epochs, "epochs_completed": len(history), "patience": args.patience,
        "best_epoch": int(checkpoint["epoch"]), "training_seconds": elapsed, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "device": str(device), "seed": args.seed, "encoder_parameter_count": sum(item.numel() for item in encoder.parameters()), "decoder_parameter_count": sum(item.numel() for item in decoder.parameters()),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": config, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
