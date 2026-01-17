import argparse
import json
import sys
from pathlib import Path

import torch
import pandas as pd
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from hv.dataset import HVDataModule
from hv.train import LitClassifier
from hv.utils import load_config, ensure_dir, set_seed
from hv.metrics import accuracy_at_threshold, f1_score_at_threshold


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


def compute_metrics(probs, targets, threshold):
    targets_int = targets.int()
    auroc = binary_auroc(probs, targets_int).item()
    auprc = binary_average_precision(probs, targets_int).item()
    acc = accuracy_at_threshold(probs, targets, threshold)
    f1 = f1_score_at_threshold(probs, targets, threshold)
    return {"auroc": auroc, "auprc": auprc, "acc": acc, "f1": f1}


def load_threshold_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt.get("best_threshold")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--external_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold_source", choices=["ckpt", "fixed"], default="ckpt")
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.data.csv_path = args.external_csv
    cfg.data.eval_external = True

    set_seed(cfg.seed)

    dm = HVDataModule(cfg, overfit_n=0)
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitClassifier.load_from_checkpoint(args.ckpt, cfg=cfg, pos_weight=dm.pos_weight)
    model.to(device)

    threshold = args.fixed_threshold
    if args.threshold_source == "ckpt":
        ckpt_threshold = load_threshold_from_ckpt(args.ckpt)
        if ckpt_threshold is not None:
            threshold = float(ckpt_threshold)
        else:
            print("[warn] best_threshold not found in checkpoint; using fixed_threshold.")

    probs, targets, image_paths = collect_predictions(model, dm.test_dataloader(), device)
    preds = (probs >= threshold).int().tolist()
    metrics = compute_metrics(probs, targets, threshold)

    n_pos = int((targets == 1).sum().item())
    n_neg = int((targets == 0).sum().item())
    metrics["threshold_used"] = float(threshold)
    metrics["n_pos"] = n_pos
    metrics["n_neg"] = n_neg

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    pred_df = pd.DataFrame(
        {
            "image_path": image_paths,
            "y_true": targets.int().tolist(),
            "y_prob": probs.tolist(),
            "y_pred": preds,
            "threshold_used": float(threshold),
        }
    )
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[external] size: {len(targets)} | good/bad: {n_pos}/{n_neg}")
    print(f"[external] threshold used: {threshold}")
    print("External metrics:")
    print(metrics)
    print(f"Predictions saved to: {pred_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
