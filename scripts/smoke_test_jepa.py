"""Run the small Phase 4 JEPA smoke test before full experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.completion_dataset import CompletionDataset
from src.jepa_completion_losses import jepa_loss, latent_variance_metrics, masked_reconstruction_bce
from src.jepa_completion_model import JEPACompletionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase4_jepa/smoke_test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    train_dataset = CompletionDataset(args.subset_root, "train", "central_block", 0.25, args.seed)
    val_dataset = CompletionDataset(args.subset_root, "val", "central_block", 0.25, args.seed)
    train_loader = DataLoader(Subset(train_dataset, range(min(500, len(train_dataset)))), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(val_dataset, range(min(50, len(val_dataset)))), batch_size=50, shuffle=False)
    model = JEPACompletionModel().to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3, weight_decay=1e-4)
    target_before = next(model.target_encoder.parameters()).detach().clone()
    context_gradients = predictor_gradients = 0
    last_stats = {}
    for epoch in range(2):
        model.train()
        model.target_encoder.eval()
        for batch in train_loader:
            inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, target)
            latent = jepa_loss(outputs["z_pred"], outputs["z_target"])
            reconstruction = masked_reconstruction_bce(outputs["logits"], target, mask)
            (latent + 0.1 * reconstruction).backward()
            context_gradients += sum(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.context_encoder.parameters())
            predictor_gradients += sum(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.predictor.parameters())
            optimizer.step()
            model.update_target_encoder()
    with torch.inference_mode():
        batch = next(iter(val_loader))
        outputs = model(batch["input"].to(device), batch["target"].to(device))
        last_stats = latent_variance_metrics(outputs["z_context"], outputs["z_target"], outputs["z_pred"])
        output_shape = list(outputs["logits"].shape)
    target_after = next(model.target_encoder.parameters()).detach().clone()
    target_grads = [parameter.grad for parameter in model.target_encoder.parameters()]
    result = {
        "device": str(device), "train_samples": len(train_loader.dataset), "val_samples": len(val_loader.dataset), "epochs": 2,
        "output_shape": output_shape, "context_finite_nonzero_gradient_checks": int(context_gradients), "predictor_finite_nonzero_gradient_checks": int(predictor_gradients),
        "target_encoder_changed_by_ema": bool(not torch.equal(target_before, target_after)), "target_encoder_has_no_grad": all(gradient is None for gradient in target_grads),
        "latent_metrics": last_stats, "all_latent_metrics_finite": all(torch.isfinite(torch.tensor(value)) for value in last_stats.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not (result["target_encoder_changed_by_ema"] and result["target_encoder_has_no_grad"] and result["output_shape"] == [50, 1, 16, 16] and result["all_latent_metrics_finite"]):
        raise RuntimeError("JEPA smoke test failed")


if __name__ == "__main__":
    main()
