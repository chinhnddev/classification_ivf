import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

from hv.dataset import HVDataModule
from hv.metrics import find_best_threshold, accuracy_at_threshold, f1_score_at_threshold
from hv.train import LitClassifier
from hv.utils import load_config, ensure_dir, set_seed


def collect_predictions(model, loader, device):
    probs_list = []
    logits_list = []
    targets_list = []
    image_paths = []
    stage_preds = []
    morph_preds = []
    stage_targets = []
    morph_targets = []
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
                logits_list.append(logits_quality.cpu())
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
                if "stage" in batch:
                    stage_targets.extend(batch["stage"].cpu().tolist())

            if collect_morph is None:
                collect_morph = logits_morph is not None
            if collect_morph:
                if logits_morph is not None:
                    morph_preds.extend(torch.argmax(logits_morph, dim=1).cpu().tolist())
                else:
                    morph_preds.extend([-1] * batch_size)
                if "morph" in batch:
                    morph_targets.extend(batch["morph"].cpu().tolist())

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
    logits = torch.cat(logits_list) if logits_list else torch.empty(0)
    targets = torch.cat(targets_list).float() if targets_list else torch.empty(0)
    stage_out = stage_preds if collect_stage else None
    morph_out = morph_preds if collect_morph else None
    exp_out = exp_preds if collect_exp else None
    icm_out = icm_preds if collect_icm else None
    te_out = te_preds if collect_te else None
    stage_true = stage_targets if collect_stage and stage_targets else None
    morph_true = morph_targets if collect_morph and morph_targets else None
    exp_true = exp_targets if collect_exp and exp_targets else None
    icm_true = icm_targets if collect_icm and icm_targets else None
    te_true = te_targets if collect_te and te_targets else None
    return (
        probs,
        targets,
        logits,
        image_paths,
        stage_out,
        morph_out,
        exp_out,
        icm_out,
        te_out,
        stage_true,
        morph_true,
        exp_true,
        icm_true,
        te_true,
    )


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


def find_temperature(val_logits, val_targets, max_iter=50):
    if val_logits.numel() == 0 or val_targets.numel() == 0:
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t)
        loss = F.binary_cross_entropy_with_logits(val_logits / t, val_targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_t).item())
    return max(temperature, 1.0)


def resolve_config(ckpt_path, config_path=None):
    if config_path:
        return load_config(config_path)
    ckpt_path = Path(ckpt_path)
    candidates = [ckpt_path.parent / "config.yaml", ckpt_path.parent.parent / "config.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return load_config(candidate)
    raise FileNotFoundError("config.yaml not found near checkpoint; pass --config explicitly.")


def run_eval(cfg, ckpt_path, threshold_source="scan", fixed_threshold=0.5, use_temp_scaling=True):
    set_seed(cfg.seed)
    dm = HVDataModule(cfg, overfit_n=0)
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitClassifier.load_from_checkpoint(
        ckpt_path,
        cfg=cfg,
        pos_weight=dm.pos_weight,
        strict=False,
    )
    model.to(device)

    val_probs, val_targets, val_logits, _, _, _, _, _, _, _, _, _, _, _ = collect_predictions(
        model, dm.val_dataloader(), device
    )
    temperature = 1.0
    if use_temp_scaling:
        temperature = find_temperature(val_logits, val_targets)
    if val_logits.numel():
        if use_temp_scaling:
            val_probs = torch.sigmoid(val_logits / temperature)
        else:
            val_probs = torch.sigmoid(val_logits)
    if val_probs.numel() == 0 or val_targets.numel() == 0:
        best_t = float(fixed_threshold)
        best_f1 = 0.0
        val_metrics = {}
    elif threshold_source == "scan":
        best_t, best_f1 = find_best_threshold(val_probs, val_targets)
        val_metrics = compute_metrics_at_threshold(val_probs, val_targets, best_t)
        val_metrics["f1"] = best_f1
    else:
        best_t = float(fixed_threshold)
        if threshold_source == "ckpt":
            ckpt_threshold = load_threshold_from_ckpt(ckpt_path)
            if ckpt_threshold is not None:
                best_t = float(ckpt_threshold)
        best_f1 = f1_score_at_threshold(val_probs, val_targets, best_t)
        val_metrics = compute_metrics_at_threshold(val_probs, val_targets, best_t)
        val_metrics["f1"] = best_f1

    (
        test_probs,
        test_targets,
        test_logits,
        test_paths,
        stage_preds,
        morph_preds,
        exp_preds,
        icm_preds,
        te_preds,
        stage_true,
        morph_true,
        exp_true,
        icm_true,
        te_true,
    ) = collect_predictions(
        model, dm.test_dataloader(), device
    )
    if test_logits.numel():
        if use_temp_scaling:
            test_probs = torch.sigmoid(test_logits / temperature)
        else:
            test_probs = torch.sigmoid(test_logits)
    test_metrics = {}
    if test_probs.numel() > 0 and test_targets.numel() > 0:
        test_metrics = compute_metrics_at_threshold(test_probs, test_targets, best_t)

    preds = (test_probs >= best_t).int().tolist() if test_probs.numel() else [-1] * len(test_paths)
    pred_data = {
        "image_path": test_paths,
        "y_true": test_targets.int().tolist() if test_targets.numel() else [-1] * len(test_paths),
        "y_prob": test_probs.tolist() if test_probs.numel() else [None] * len(test_paths),
        "y_pred": preds,
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

    if stage_preds is not None and stage_true is not None:
        stage_true_t = torch.tensor(stage_true)
        stage_pred_t = torch.tensor(stage_preds)
        valid = stage_true_t != -1
        if valid.any():
            test_metrics["stage_acc"] = float((stage_pred_t[valid] == stage_true_t[valid]).float().mean().item())

    if morph_preds is not None and morph_true is not None:
        morph_true_t = torch.tensor(morph_true)
        morph_pred_t = torch.tensor(morph_preds)
        valid = morph_true_t != -1
        if valid.any():
            test_metrics["morph_acc"] = float((morph_pred_t[valid] == morph_true_t[valid]).float().mean().item())

    if exp_preds is not None and exp_true is not None:
        exp_true_t = torch.tensor(exp_true)
        exp_pred_t = torch.tensor(exp_preds)
        valid = exp_true_t != -1
        if valid.any():
            test_metrics["exp_acc"] = float((exp_pred_t[valid] == exp_true_t[valid]).float().mean().item())

    if icm_preds is not None and icm_true is not None:
        icm_true_t = torch.tensor(icm_true)
        icm_pred_t = torch.tensor(icm_preds)
        valid = icm_true_t != -1
        if valid.any():
            test_metrics["icm_acc"] = float((icm_pred_t[valid] == icm_true_t[valid]).float().mean().item())

    if te_preds is not None and te_true is not None:
        te_true_t = torch.tensor(te_true)
        te_pred_t = torch.tensor(te_preds)
        valid = te_true_t != -1
        if valid.any():
            test_metrics["te_acc"] = float((te_pred_t[valid] == te_true_t[valid]).float().mean().item())

    output_dir = Path(cfg.logging.output_dir)
    ensure_dir(output_dir)
    preds_path = output_dir / "predictions.csv"
    pred_df.to_csv(preds_path, index=False)

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "best_threshold": best_t,
        "temperature": temperature,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Validation metrics:")
    print(val_metrics)
    if use_temp_scaling:
        print(f"Temperature: {temperature}")
    else:
        print("Temperature scaling disabled (T=1.0)")
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
    parser.add_argument("--disable_temperature_scaling", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config(args.ckpt, args.config)
    run_eval(
        cfg,
        args.ckpt,
        threshold_source=args.threshold_source,
        fixed_threshold=args.fixed_threshold,
        use_temp_scaling=not args.disable_temperature_scaling,
    )


if __name__ == "__main__":
    main()
