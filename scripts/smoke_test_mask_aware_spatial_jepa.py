"""Run the Phase 4.2 mask-aware spatial-JEPA smoke test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.completion_dataset import CompletionDataset
from src.mask_aware_spatial_jepa_losses import downsample_mask, mask_aware_spatial_jepa_loss, mask_weight_map, masked_reconstruction_bce, spatial_jepa_distance, spatial_latent_statistics
from src.spatial_jepa_completion_model import SpatialJEPACompletionModel, compose_binary_spatial_completion


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase4_2/mask_aware_smoke"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    device = resolve_device(args.device)
    train_dataset = CompletionDataset(args.subset_root, "train", "central_block", 0.25, 42)
    val_dataset = CompletionDataset(args.subset_root, "val", "central_block", 0.25, 42)
    train_loader = DataLoader(Subset(train_dataset, range(500)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(val_dataset, range(50)), batch_size=50, shuffle=False)
    model = SpatialJEPACompletionModel().to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3, weight_decay=1e-4)
    initial_target = {name: value.detach().cpu().clone() for name, value in model.target_encoder.state_dict().items()}
    context_checks = predictor_checks = decoder_checks = 0
    gradient_ratio = None
    for epoch in range(args.epochs):
        model.train()
        model.target_encoder.eval()
        for batch_index, batch in enumerate(train_loader):
            inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, target)
            mask8 = downsample_mask(mask)
            weights = mask_weight_map(mask, args.alpha)
            latent = mask_aware_spatial_jepa_loss(outputs["z_pred"], outputs["z_target"], weights)
            reconstruction = masked_reconstruction_bce(outputs["logits"], target, mask)
            total = latent + 0.1 * reconstruction
            assert all(torch.isfinite(value) for value in (latent, reconstruction, total))
            if epoch == 0 and batch_index == 0:
                uniform_gradient = torch.autograd.grad(spatial_jepa_distance(outputs["z_pred"], outputs["z_target"]).mean(), outputs["z_pred"], retain_graph=True)[0]
                weighted_gradient = torch.autograd.grad((spatial_jepa_distance(outputs["z_pred"], outputs["z_target"]) * weights[:, 0]).sum() / weights.sum(), outputs["z_pred"], retain_graph=True)[0]
                ratio = (weighted_gradient.abs() / uniform_gradient.abs().clamp_min(1e-8)).mean(dim=1)
                high = mask8[:, 0] > args.alpha * 0 + 0.5
                low = mask8[:, 0] == 0
                gradient_ratio = {"uniform_mean": float(uniform_gradient.abs().mean().item()), "weighted_mean": float(weighted_gradient.abs().mean().item()), "high_weight_relative_gradient": float(ratio[high].mean().item()), "low_weight_relative_gradient": float(ratio[low].mean().item()), "high_to_low_relative_gradient": float((ratio[high].mean() / ratio[low].mean()).item())}
            total.backward()
            context_checks += sum(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.context_encoder.parameters())
            predictor_checks += sum(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.predictor.parameters())
            decoder_checks += sum(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.decoder.parameters())
            assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
            optimizer.step()
            model.update_target_encoder()
    target_changed = any(not torch.equal(initial_target[name], value.detach().cpu()) for name, value in model.target_encoder.state_dict().items())
    with torch.inference_mode():
        validation = next(iter(val_loader))
        inputs, target, mask = validation["input"].to(device), validation["target"].to(device), validation["mask"].to(device)
        outputs = model(inputs, target)
        weights = mask_weight_map(mask, args.alpha)
        probabilities = torch.sigmoid(outputs["logits"])
        completed = compose_binary_spatial_completion(probabilities, inputs, mask)
        known_error = torch.abs(completed - target) * (1.0 - mask)
        latent_metrics = spatial_latent_statistics(outputs["z_context"].cpu(), outputs["z_target"].cpu(), outputs["z_pred"].cpu())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "smoke_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": args.epochs, "alpha": args.alpha}, checkpoint)
    result = {
        "device": str(device), "train_samples": 500, "val_samples": 50, "epochs": args.epochs, "alpha": args.alpha,
        "mask8_shape": list(mask8.shape), "mask8_min": float(mask8.min().item()), "mask8_max": float(mask8.max().item()), "weight_min": float(weights.min().item()), "weight_max": float(weights.max().item()),
        "masked_jepa_finite": True, "reconstruction_finite": True, "total_finite": True,
        "context_finite_nonzero_gradient_checks": int(context_checks), "predictor_finite_nonzero_gradient_checks": int(predictor_checks), "decoder_finite_nonzero_gradient_checks": int(decoder_checks), "gradient_weighting": gradient_ratio,
        "target_encoder_changed_by_ema": bool(target_changed), "target_encoder_has_no_grad": bool(all(parameter.grad is None for parameter in model.target_encoder.parameters())), "known_region_error": float(known_error.max().item()), "latent_metrics": latent_metrics, "latent_metrics_finite": bool(all(torch.isfinite(torch.tensor(value)) for value in latent_metrics.values())), "checkpoint": str(checkpoint),
    }
    with (args.output_dir / "mask_weight_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mean_mask8", "min_mask8", "max_mask8", "mean_weight", "min_weight", "max_weight"))
        writer.writerow((float(mask8.mean().item()), float(mask8.min().item()), float(mask8.max().item()), float(weights.mean().item()), float(weights.min().item()), float(weights.max().item())))
    (args.output_dir / "smoke_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
