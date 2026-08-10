"""Run the Phase 4.1 spatial-JEPA smoke test before full experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.completion_dataset import CompletionDataset
from src.spatial_jepa_completion_losses import masked_reconstruction_bce, spatial_jepa_loss, spatial_latent_statistics
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase4_1/spatial_smoke"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    device = resolve_device(args.device)
    train_dataset = CompletionDataset(args.subset_root, "train", "central_block", 0.25, 42)
    val_dataset = CompletionDataset(args.subset_root, "val", "central_block", 0.25, 42)
    train_loader = DataLoader(Subset(train_dataset, range(500)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(val_dataset, range(50)), batch_size=50, shuffle=False)
    model = SpatialJEPACompletionModel().to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3, weight_decay=1e-4)
    initial_target = {name: value.detach().cpu().clone() for name, value in model.target_encoder.state_dict().items()}
    last_outputs: dict[str, torch.Tensor] | None = None
    context_checks = predictor_checks = decoder_checks = 0
    for _ in range(args.epochs):
        model.train()
        model.target_encoder.eval()
        for batch in train_loader:
            inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, target)
            loss = spatial_jepa_loss(outputs["z_pred"], outputs["z_target"]) + 0.1 * masked_reconstruction_bce(outputs["logits"], target, mask)
            assert torch.isfinite(loss)
            loss.backward()
            context_checks += sum(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.context_encoder.parameters())
            predictor_checks += sum(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.predictor.parameters())
            decoder_checks += sum(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.decoder.parameters())
            assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
            optimizer.step()
            model.update_target_encoder()
            last_outputs = outputs
    assert last_outputs is not None
    target_changed = any(not torch.equal(initial_target[name], value.detach().cpu()) for name, value in model.target_encoder.state_dict().items())
    with torch.inference_mode():
        validation = next(iter(val_loader))
        outputs = model(validation["input"].to(device), validation["target"].to(device))
        probabilities = torch.sigmoid(outputs["logits"])
        completed = compose_binary_spatial_completion(probabilities, validation["input"].to(device), validation["mask"].to(device))
        known_error = torch.abs(completed - validation["target"].to(device)) * (1.0 - validation["mask"].to(device))
        latent_metrics = spatial_latent_statistics(outputs["z_context"].cpu(), outputs["z_target"].cpu(), outputs["z_pred"].cpu())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "smoke_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": args.epochs}, checkpoint)
    result = {
        "device": str(device), "train_samples": 500, "val_samples": 50, "epochs": args.epochs,
        "context_shape": list(outputs["z_context"].shape), "target_shape": list(outputs["z_target"].shape), "predictor_shape": list(outputs["z_pred"].shape), "decoder_shape": list(outputs["logits"].shape),
        "context_finite_nonzero_gradient_checks": int(context_checks), "predictor_finite_nonzero_gradient_checks": int(predictor_checks), "decoder_finite_nonzero_gradient_checks": int(decoder_checks),
        "target_encoder_changed_by_ema": bool(target_changed), "target_encoder_has_no_grad": bool(all(parameter.grad is None for parameter in model.target_encoder.parameters())),
        "known_region_error": float(known_error.max().item()), "latent_metrics": latent_metrics, "latent_metrics_finite": bool(all(torch.isfinite(torch.tensor(value)) for value in latent_metrics.values())), "checkpoint": str(checkpoint),
    }
    (args.output_dir / "smoke_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
