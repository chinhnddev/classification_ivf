import torch
from torchmetrics.functional.classification import binary_auroc, binary_average_precision


def accuracy_at_threshold(probs, targets, threshold):
    preds = (probs >= threshold).float()
    acc = (preds == targets).float().mean().item()
    return acc


def f1_score_at_threshold(probs, targets, threshold):
    preds = (probs >= threshold).float()
    targets = targets.float()
    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    denom = (2.0 * tp + fp + fn)
    if denom.item() == 0:
        return 0.0
    return (2.0 * tp / denom).item()


def find_best_threshold(probs, targets, num_thresholds=101):
    best_t = 0.5
    best_f1 = -1.0
    for t in torch.linspace(0.0, 1.0, num_thresholds):
        f1 = f1_score_at_threshold(probs, targets, t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t.item())
    return float(best_t), float(best_f1)


def compute_metrics(probs, targets, best_threshold):
    targets = targets.int()
    auroc = binary_auroc(probs, targets).item()
    auprc = binary_average_precision(probs, targets).item()
    acc = accuracy_at_threshold(probs, targets, 0.5)
    f1 = f1_score_at_threshold(probs, targets, best_threshold)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "acc": acc,
        "f1": f1,
    }
