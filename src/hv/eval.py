import argparse
import json
from pathlib import Path
import torch
import pandas as pd
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

from hv.dataset import HVDataModule
from hv.metrics import find_best_threshold, accuracy_at_threshold, f1_score_at_threshold
from hv.train import LitClassifier
from hv.utils import load_config, ensure_dir, set_seed


def collect_predictions(model, loader, device):
    probs_list = []
    targets_list = []
    image_paths = []
    stage_preds = []
    morph_preds = []
    collect_stage = None
    collect_morph = None
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                logits_quality = outputs["logits_quality"]
                logits_stage = outputs.get("logits_stage")
                logits_morph = outputs.get("logits_morph")
            else:
                logits_quality = outputs
                logits_stage = None
                logits_morph = None
            probs = torch.sigmoid(logits_quality)
            probs_list.append(probs.cpu())
            targets_list.append(batch["label"].cpu())
            image_paths.extend(batch["image_path"])

            batch_size = probs.shape[0]
            if collect_stage is None:
                collect_stage = logits_stage is not None
            if collect_stage:
                if logits_stage is not None:
                    stage_preds.extend(torch.argmax(logits_stage, dim=1).cpu().tolist())
                else:
                    stage_preds.extend([-1] * batch_size)

            if collect_morph is None:
                collect_morph = logits_morph is not None
            if collect_morph:
                if logits_morph is not None:
                    morph_preds.extend(torch.argmax(logits_morph, dim=1).cpu().tolist())
                else:
                    morph_preds.extend([-1] * batch_size)

    probs = torch.cat(probs_list) if probs_list else torch.empty(0)
    targets = torch.cat(targets_list).float() if targets_list else torch.empty(0)
    stage_out = stage_preds if collect_stage else None
    morph_out = morph_preds if collect_morph else None
    return probs, targets, image_paths, stage_out, morph_out


def compute_metrics_at_threshold(probs, targets, threshold):
    targets_int = targets.int()
    auroc = binary_auroc(probs, targets_int).item()
    auprc = binary_average_precision(probs, targets_int).item()
    acc = accuracy_at_threshold(probs, targets, threshold)
    f1 = f1_score_at_threshold(probs, targets, threshold)
    return {"auroc": auroc, "auprc": auprc, "acc": acc, "f1": f1}


def load_threshold_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt.get("best_threshold")


def resolve_config(ckpt_path, config_path=None):
    if config_path:
        return load_config(config_path)
    ckpt_path = Path(ckpt_path)
    candidates = [ckpt_path.parent / "config.yaml", ckpt_path.parent.parent / "config.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return load_config(candidate)
    raise FileNotFoundError("config.yaml not found near checkpoint; pass --config explicitly.")


def run_eval(cfg, ckpt_path, threshold_source="scan", fixed_threshold=0.5):
    set_seed(cfg.seed)
    dm = HVDataModule(cfg, overfit_n=0)
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitClassifier.load_from_checkpoint(ckpt_path, cfg=cfg, pos_weight=dm.pos_weight)
    model.to(device)

    val_probs, val_targets, _, _, _ = collect_predictions(model, dm.val_dataloader(), device)
    if threshold_source == "scan":
        best_t, best_f1 = find_best_threshold(val_probs, val_targets)
    else:
        best_t = float(fixed_threshold)
        if threshold_source == "ckpt":
            ckpt_threshold = load_threshold_from_ckpt(ckpt_path)
            if ckpt_threshold is not None:
                best_t = float(ckpt_threshold)
        best_f1 = f1_score_at_threshold(val_probs, val_targets, best_t)

    val_metrics = compute_metrics_at_threshold(val_probs, val_targets, best_t)
    val_metrics["f1"] = best_f1

    test_probs, test_targets, test_paths, stage_preds, morph_preds = collect_predictions(
        model, dm.test_dataloader(), device
    )
    test_metrics = compute_metrics_at_threshold(test_probs, test_targets, best_t)

    preds = (test_probs >= best_t).int().tolist()
    pred_data = {
        "image_path": test_paths,
        "y_true": test_targets.int().tolist(),
        "y_prob": test_probs.tolist(),
        "y_pred": preds,
    }
    if stage_preds is not None:
        pred_data["stage_pred"] = stage_preds
    if morph_preds is not None:
        pred_data["morph_pred"] = morph_preds
    pred_df = pd.DataFrame(pred_data)

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
    parser.add_argument(
        "--threshold_source",
        choices=["scan", "ckpt", "fixed"],
        default="scan",
    )
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config(args.ckpt, args.config)
    run_eval(cfg, args.ckpt, threshold_source=args.threshold_source, fixed_threshold=args.fixed_threshold)


if __name__ == "__main__":
    main()
