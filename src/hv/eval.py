import argparse
import json
from pathlib import Path
import torch
import pandas as pd

from hv.dataset import HVDataModule
from hv.metrics import find_best_threshold, compute_metrics
from hv.train import LitClassifier
from hv.utils import load_config, ensure_dir, set_seed


def collect_predictions(model, loader, device):
    probs_list = []
    targets_list = []
    image_paths = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu())
            targets_list.append(batch["label"].cpu())
            image_paths.extend(batch["image_path"])
    probs = torch.cat(probs_list) if probs_list else torch.empty(0)
    targets = torch.cat(targets_list).float() if targets_list else torch.empty(0)
    return probs, targets, image_paths


def resolve_config(ckpt_path, config_path=None):
    if config_path:
        return load_config(config_path)
    ckpt_path = Path(ckpt_path)
    candidates = [ckpt_path.parent / "config.yaml", ckpt_path.parent.parent / "config.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return load_config(candidate)
    raise FileNotFoundError("config.yaml not found near checkpoint; pass --config explicitly.")


def run_eval(cfg, ckpt_path):
    set_seed(cfg.seed)
    dm = HVDataModule(cfg, overfit_n=0)
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitClassifier.load_from_checkpoint(ckpt_path, cfg=cfg, pos_weight=dm.pos_weight)
    model.to(device)

    val_probs, val_targets, _ = collect_predictions(model, dm.val_dataloader(), device)
    best_t, best_f1 = find_best_threshold(val_probs, val_targets)
    val_metrics = compute_metrics(val_probs, val_targets, best_t)
    val_metrics["f1"] = best_f1

    test_probs, test_targets, test_paths = collect_predictions(model, dm.test_dataloader(), device)
    test_metrics = compute_metrics(test_probs, test_targets, best_t)

    preds = (test_probs >= best_t).int().tolist()
    pred_df = pd.DataFrame(
        {
            "image_path": test_paths,
            "y_true": test_targets.int().tolist(),
            "y_prob": test_probs.tolist(),
            "y_pred": preds,
        }
    )

    output_dir = Path(cfg.logging.output_dir)
    ensure_dir(output_dir)
    preds_path = output_dir / "predictions.csv"
    pred_df.to_csv(preds_path, index=False)

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "best_threshold": best_t,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Validation metrics:")
    print(val_metrics)
    print(f"Best threshold: {best_t}")
    print("Test metrics:")
    print(test_metrics)
    print(f"Predictions saved to: {preds_path}")
    print(f"Metrics saved to: {metrics_path}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config(args.ckpt, args.config)
    run_eval(cfg, args.ckpt)


if __name__ == "__main__":
    main()
