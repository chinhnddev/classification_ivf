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
    stage_preds = []
    morph_preds = []
    exp_preds = []
    icm_preds = []
    te_preds = []
    exp_targets = []
    icm_targets = []
    te_targets = []
    collect_stage = None
    collect_morph = None
    collect_exp = None
    collect_icm = None
    collect_te = None
    model.eval()
    probs = torch.empty(0)
    batch_size = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                logits_quality = outputs["logits_quality"]
                logits_stage = outputs.get("logits_stage")
                logits_morph = outputs.get("logits_morph")
                logits_exp = outputs.get("logits_exp")
                logits_icm = outputs.get("logits_icm")
                logits_te = outputs.get("logits_te")
            else:
                logits_quality = outputs
                logits_stage = None
                logits_morph = None
                logits_exp = None
                logits_icm = None
                logits_te = None
            probs = torch.empty(0)
            if logits_quality is not None:
                probs = torch.sigmoid(logits_quality)
                probs_list.append(probs.cpu())
                targets_list.append(batch["label"].cpu())
            image_paths.extend(batch["image_path"])

            batch_size = len(batch["image_path"])
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

            if collect_exp is None:
                collect_exp = logits_exp is not None
            if collect_exp:
                if logits_exp is not None:
                    exp_preds.extend(torch.argmax(logits_exp, dim=1).cpu().tolist())
                else:
                    exp_preds.extend([-1] * batch_size)
                if "exp" in batch:
                    exp_targets.extend(batch["exp"].cpu().tolist())

            if collect_icm is None:
                collect_icm = logits_icm is not None
            if collect_icm:
                if logits_icm is not None:
                    icm_preds.extend(torch.argmax(logits_icm, dim=1).cpu().tolist())
                else:
                    icm_preds.extend([-1] * batch_size)
                if "icm" in batch:
                    icm_targets.extend(batch["icm"].cpu().tolist())

            if collect_te is None:
                collect_te = logits_te is not None
            if collect_te:
                if logits_te is not None:
                    te_preds.extend(torch.argmax(logits_te, dim=1).cpu().tolist())
                else:
                    te_preds.extend([-1] * batch_size)
                if "te" in batch:
                    te_targets.extend(batch["te"].cpu().tolist())

    probs = torch.cat(probs_list) if probs_list else torch.empty(0)
    targets = torch.cat(targets_list).float() if targets_list else torch.empty(0)
    stage_out = stage_preds if collect_stage else None
    morph_out = morph_preds if collect_morph else None
    exp_out = exp_preds if collect_exp else None
    icm_out = icm_preds if collect_icm else None
    te_out = te_preds if collect_te else None
    exp_true = exp_targets if collect_exp and exp_targets else None
    icm_true = icm_targets if collect_icm and icm_targets else None
    te_true = te_targets if collect_te and te_targets else None
    return (
        probs,
        targets,
        image_paths,
        stage_out,
        morph_out,
        exp_out,
        icm_out,
        te_out,
        exp_true,
        icm_true,
        te_true,
    )


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

    probs, targets, image_paths, stage_preds, morph_preds, exp_preds, icm_preds, te_preds, exp_true, icm_true, te_true = collect_predictions(
        model, dm.test_dataloader(), device
    )
    preds = (probs >= threshold).int().tolist() if probs.numel() else []
    metrics = {}
    if probs.numel() and targets.numel():
        metrics = compute_metrics(probs, targets, threshold)

    n_pos = int((targets == 1).sum().item()) if targets.numel() else 0
    n_neg = int((targets == 0).sum().item()) if targets.numel() else 0
    metrics["threshold_used"] = float(threshold)
    metrics["n_pos"] = n_pos
    metrics["n_neg"] = n_neg

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    pred_data = {
        "image_path": image_paths,
        "y_true": targets.int().tolist() if targets.numel() else [-1] * len(image_paths),
        "y_prob": probs.tolist() if probs.numel() else [None] * len(image_paths),
        "y_pred": preds,
        "threshold_used": float(threshold),
    }
    if stage_preds is not None:
        pred_data["stage_pred"] = stage_preds
    if morph_preds is not None:
        pred_data["morph_pred"] = morph_preds
    if exp_preds is not None:
        pred_data["exp_pred"] = exp_preds
    if icm_preds is not None:
        pred_data["icm_pred"] = icm_preds
    if te_preds is not None:
        pred_data["te_pred"] = te_preds
    pred_df = pd.DataFrame(pred_data)
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    if exp_preds is not None and exp_true is not None:
        exp_true_t = torch.tensor(exp_true)
        exp_pred_t = torch.tensor(exp_preds)
        valid = exp_true_t != -1
        if valid.any():
            metrics["exp_acc"] = float((exp_pred_t[valid] == exp_true_t[valid]).float().mean().item())

    if icm_preds is not None and icm_true is not None:
        icm_true_t = torch.tensor(icm_true)
        icm_pred_t = torch.tensor(icm_preds)
        valid = icm_true_t != -1
        if valid.any():
            metrics["icm_acc"] = float((icm_pred_t[valid] == icm_true_t[valid]).float().mean().item())

    if te_preds is not None and te_true is not None:
        te_true_t = torch.tensor(te_true)
        te_pred_t = torch.tensor(te_preds)
        valid = te_true_t != -1
        if valid.any():
            metrics["te_acc"] = float((te_pred_t[valid] == te_true_t[valid]).float().mean().item())

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
