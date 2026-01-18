import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision

from hv.dataset import HVDataModule
from hv.models import build_model
from hv.metrics import find_best_threshold
from hv.utils import set_seed, load_config, ensure_dir, save_config


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=1.5, gamma_neg=0.5, alpha=0.75, eps=1e-8):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, logits, targets):
        targets = targets.float()
        prob = torch.sigmoid(logits)
        pos_loss = -self.alpha * (1.0 - prob).pow(self.gamma_pos) * F.logsigmoid(logits)
        neg_loss = -(1.0 - self.alpha) * prob.pow(self.gamma_neg) * F.logsigmoid(-logits)
        loss = pos_loss * targets + neg_loss * (1.0 - targets)
        return loss.mean()


class LitClassifier(pl.LightningModule):
    def __init__(self, cfg, pos_weight=1.0):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.loss_fn = AsymmetricFocalLoss(gamma_pos=1.5, gamma_neg=0.5, alpha=0.75)

        self.train_acc = BinaryAccuracy(threshold=0.5)
        self.val_acc = BinaryAccuracy(threshold=0.5)
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.val_probs = []
        self.val_targets = []
        self.best_threshold = 0.5
        model_cfg = _cfg_get(cfg, "model", cfg)
        self.enable_quality = bool(_cfg_get(model_cfg, "enable_quality", True))
        self.enable_stage = bool(_cfg_get(model_cfg, "enable_stage", False))
        self.enable_morph = bool(_cfg_get(model_cfg, "enable_morph", False))
        weights_cfg = _cfg_get(_cfg_get(cfg, "training", cfg), "loss_weights", None)
        self.loss_w_quality = float(_cfg_get(weights_cfg, "quality", 1.0))
        self.loss_w_stage = float(_cfg_get(weights_cfg, "stage", 0.3))
        self.loss_w_morph = float(_cfg_get(weights_cfg, "morph", 0.3))
        self.loss_w_exp = float(_cfg_get(weights_cfg, "exp", 1.0))
        self.loss_w_icm = float(_cfg_get(weights_cfg, "icm", 1.0))
        self.loss_w_te = float(_cfg_get(weights_cfg, "te", 1.0))
        training_cfg = _cfg_get(cfg, "training", cfg)
        self.task = str(_cfg_get(training_cfg, "task", "quality")).lower()
        self.freeze_encoder_ratio = float(_cfg_get(training_cfg, "freeze_encoder_ratio", 0.0))
        self.unfreeze_epochs = int(_cfg_get(training_cfg, "unfreeze_epochs", 0))
        self.initial_freeze_ratio = self.freeze_encoder_ratio
        self.loss_stage_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_morph_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_exp_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_icm_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_te_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.stage_correct = 0
        self.stage_total = 0
        self.morph_correct = 0
        self.morph_total = 0
        self.exp_correct = 0
        self.exp_total = 0
        self.icm_correct = 0
        self.icm_total = 0
        self.te_correct = 0
        self.te_total = 0
        self._apply_freeze_policy()
        self.distill_cfg = _cfg_get(cfg, "distill", None)
        self.distill_enabled = bool(_cfg_get(self.distill_cfg, "enabled", False))
        self.distill_temperature = float(_cfg_get(self.distill_cfg, "temperature", 1.0))
        self.distill_lambda = float(_cfg_get(self.distill_cfg, "lambda_kd", 0.0))
        self.teacher = None
        if self.distill_enabled:
            self.teacher = self._load_teacher(self.distill_cfg)

    def forward(self, x):
        return self.model(x)

    def _apply_freeze_policy(self):
        if self.task == "morphology":
            self._freeze_if_exists(["stage_head", "quality_head"])
        elif self.task == "stage":
            self._freeze_if_exists(["exp_head", "icm_head", "te_head", "morph_head", "quality_head"])
        elif self.task == "quality":
            self._freeze_if_exists(["stage_head", "exp_head", "icm_head", "te_head", "morph_head"])
        if self.task == "stage" and self.freeze_encoder_ratio > 0:
            self._set_encoder_freeze_ratio(self.freeze_encoder_ratio)

        model_cfg = _cfg_get(self.cfg, "model", self.cfg)
        if bool(_cfg_get(model_cfg, "freeze_stage_head", False)):
            self._freeze_if_exists(["stage_head"])
        if bool(_cfg_get(model_cfg, "freeze_morph_head", False)):
            self._freeze_if_exists(["morph_head", "exp_head", "icm_head", "te_head"])
        if bool(_cfg_get(model_cfg, "freeze_encoder", False)):
            self._freeze_if_exists(["encoder"])

        if bool(_cfg_get(model_cfg, "freeze_stage_head", False)):
            print("[freeze] stage head frozen")
        if bool(_cfg_get(model_cfg, "freeze_morph_head", False)):
            print("[freeze] morph heads frozen")
        if bool(_cfg_get(model_cfg, "freeze_encoder", False)):
            print("[freeze] encoder frozen")

    def _freeze_if_exists(self, names):
        for name in names:
            module = getattr(self.model, name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def _set_encoder_freeze_ratio(self, ratio):
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            return
        params = list(encoder.parameters())
        if not params:
            return
        n_freeze = int(len(params) * ratio)
        for idx, param in enumerate(params):
            param.requires_grad = idx >= n_freeze

    def _load_teacher(self, distill_cfg):
        teacher_name = str(_cfg_get(distill_cfg, "teacher_model", "resnet50_baseline"))
        teacher_pretrained = bool(_cfg_get(distill_cfg, "teacher_pretrained", False))
        teacher_cfg = {"model": {"name": teacher_name, "pretrained": teacher_pretrained, "dropout": 0.0}}
        teacher = build_model(teacher_cfg)
        ckpt_path = _cfg_get(distill_cfg, "teacher_ckpt", None)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            state_dict = state.get("state_dict", state)
            cleaned = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    key = key[len("model."):]
                cleaned[key] = value
            missing, unexpected = teacher.load_state_dict(cleaned, strict=False)
            if missing:
                print(f"[distill] teacher missing keys: {len(missing)}")
            if unexpected:
                print(f"[distill] teacher unexpected keys: {len(unexpected)}")
        else:
            print("[distill] teacher_ckpt not set; using teacher weights as initialized.")
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def _kd_loss(self, student_logits, teacher_logits):
        temperature = self.distill_temperature
        student_2 = torch.stack([torch.zeros_like(student_logits), student_logits], dim=1) / temperature
        teacher_2 = torch.stack([torch.zeros_like(teacher_logits), teacher_logits], dim=1) / temperature
        return F.kl_div(
            F.log_softmax(student_2, dim=1),
            F.softmax(teacher_2, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)

    def on_fit_start(self):
        if self.teacher is not None:
            self.teacher.to(self.device)

    def on_train_epoch_start(self):
        if self.task != "stage":
            return
        if self.unfreeze_epochs <= 0:
            return
        ratio = max(self.initial_freeze_ratio * (1.0 - self.current_epoch / self.unfreeze_epochs), 0.0)
        self._set_encoder_freeze_ratio(ratio)

    def _unpack_outputs(self, outputs):
        if isinstance(outputs, dict):
            return {
                "logits_quality": outputs.get("logits_quality"),
                "logits_stage": outputs.get("logits_stage"),
                "logits_morph": outputs.get("logits_morph"),
                "logits_exp": outputs.get("logits_exp"),
                "logits_icm": outputs.get("logits_icm"),
                "logits_te": outputs.get("logits_te"),
            }
        return {
            "logits_quality": outputs,
            "logits_stage": None,
            "logits_morph": None,
            "logits_exp": None,
            "logits_icm": None,
            "logits_te": None,
        }

    def training_step(self, batch, batch_idx):
        outputs = self(batch["image"])
        outputs = self._unpack_outputs(outputs)
        logits = outputs["logits_quality"]
        loss = 0.0
        batch_size = batch["image"].shape[0]

        if self.task in {"quality", "all"} and logits is not None and "label" in batch:
            targets = batch["label"].float()
            loss_quality = self.loss_fn(logits, targets)
            loss = loss + self.loss_w_quality * loss_quality
            probs = torch.sigmoid(logits)
            self.train_acc.update(probs, targets.int())
            if self.distill_enabled and self.teacher is not None and self.distill_lambda > 0:
                with torch.no_grad():
                    teacher_out = self.teacher(batch["image"])
                teacher_logits = self._unpack_outputs(teacher_out)["logits_quality"]
                if teacher_logits is not None:
                    kd_loss = self._kd_loss(logits, teacher_logits)
                    loss = loss + self.distill_lambda * kd_loss
                    self.log("train_kd_loss", kd_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        if self.task in {"stage", "all"} and self.enable_stage and outputs["logits_stage"] is not None and "stage" in batch:
            stage_targets = batch["stage"].long()
            if (stage_targets != -1).any():
                loss_stage = self.loss_stage_fn(outputs["logits_stage"], stage_targets)
                loss = loss + self.loss_w_stage * loss_stage
            else:
                if batch_idx == 0:
                    print("[warn] stage targets are all -1; stage loss skipped.")
        if self.task in {"morphology", "all"} and self.enable_morph:
            if outputs["logits_morph"] is not None and "morph" in batch:
                morph_targets = batch["morph"].long()
                if (morph_targets != -1).any():
                    loss_morph = self.loss_morph_fn(outputs["logits_morph"], morph_targets)
                    loss = loss + self.loss_w_morph * loss_morph
                else:
                    if batch_idx == 0:
                        print("[warn] morph targets are all -1; morph loss skipped.")
            if outputs["logits_exp"] is not None and "exp" in batch:
                exp_targets = batch["exp"].long()
                if (exp_targets != -1).any():
                    loss_exp = self.loss_exp_fn(outputs["logits_exp"], exp_targets)
                    loss = loss + self.loss_w_exp * loss_exp
                else:
                    if batch_idx == 0:
                        print("[warn] exp targets are all -1; exp loss skipped.")
            if outputs["logits_icm"] is not None and "icm" in batch:
                icm_targets = batch["icm"].long()
                if (icm_targets != -1).any():
                    loss_icm = self.loss_icm_fn(outputs["logits_icm"], icm_targets)
                    loss = loss + self.loss_w_icm * loss_icm
                else:
                    if batch_idx == 0:
                        print("[warn] icm targets are all -1; icm loss skipped.")
            if outputs["logits_te"] is not None and "te" in batch:
                te_targets = batch["te"].long()
                if (te_targets != -1).any():
                    loss_te = self.loss_te_fn(outputs["logits_te"], te_targets)
                    loss = loss + self.loss_w_te * loss_te
                else:
                    if batch_idx == 0:
                        print("[warn] te targets are all -1; te loss skipped.")

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(float(loss), device=self.device)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if self.task in {"quality", "all"} and logits is not None and "label" in batch:
            self.log(
                "train_acc",
                self.train_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["image"])
        outputs = self._unpack_outputs(outputs)
        logits = outputs["logits_quality"]
        loss = 0.0
        if self.task in {"quality", "all"} and logits is not None and "label" in batch:
            targets = batch["label"].float()
            loss_quality = self.loss_fn(logits, targets)
            loss = loss + self.loss_w_quality * loss_quality
            probs = torch.sigmoid(logits)
            self.val_acc.update(probs, targets.int())
            self.val_auroc.update(probs, targets.int())
            self.val_auprc.update(probs, targets.int())
            self.val_probs.append(probs.detach().cpu())
            self.val_targets.append(targets.detach().cpu())
        if self.task in {"stage", "all"} and self.enable_stage and outputs["logits_stage"] is not None and "stage" in batch:
            stage_targets = batch["stage"].long()
            if (stage_targets != -1).any():
                loss_stage = self.loss_stage_fn(outputs["logits_stage"], stage_targets)
                loss = loss + self.loss_w_stage * loss_stage
                stage_preds = torch.argmax(outputs["logits_stage"], dim=1)
                valid = stage_targets != -1
                self.stage_correct += int((stage_preds[valid] == stage_targets[valid]).sum().item())
                self.stage_total += int(valid.sum().item())
            else:
                if batch_idx == 0:
                    print("[warn] stage targets are all -1; stage loss skipped.")
        if self.task in {"morphology", "all"} and self.enable_morph:
            if outputs["logits_morph"] is not None and "morph" in batch:
                morph_targets = batch["morph"].long()
                if (morph_targets != -1).any():
                    loss_morph = self.loss_morph_fn(outputs["logits_morph"], morph_targets)
                    loss = loss + self.loss_w_morph * loss_morph
                    morph_preds = torch.argmax(outputs["logits_morph"], dim=1)
                    valid = morph_targets != -1
                    self.morph_correct += int((morph_preds[valid] == morph_targets[valid]).sum().item())
                    self.morph_total += int(valid.sum().item())
                else:
                    if batch_idx == 0:
                        print("[warn] morph targets are all -1; morph loss skipped.")
            if outputs["logits_exp"] is not None and "exp" in batch:
                exp_targets = batch["exp"].long()
                if (exp_targets != -1).any():
                    loss_exp = self.loss_exp_fn(outputs["logits_exp"], exp_targets)
                    loss = loss + self.loss_w_exp * loss_exp
                    exp_preds = torch.argmax(outputs["logits_exp"], dim=1)
                    valid = exp_targets != -1
                    self.exp_correct += int((exp_preds[valid] == exp_targets[valid]).sum().item())
                    self.exp_total += int(valid.sum().item())
                else:
                    if batch_idx == 0:
                        print("[warn] exp targets are all -1; exp loss skipped.")
            if outputs["logits_icm"] is not None and "icm" in batch:
                icm_targets = batch["icm"].long()
                if (icm_targets != -1).any():
                    loss_icm = self.loss_icm_fn(outputs["logits_icm"], icm_targets)
                    loss = loss + self.loss_w_icm * loss_icm
                    icm_preds = torch.argmax(outputs["logits_icm"], dim=1)
                    valid = icm_targets != -1
                    self.icm_correct += int((icm_preds[valid] == icm_targets[valid]).sum().item())
                    self.icm_total += int(valid.sum().item())
                else:
                    if batch_idx == 0:
                        print("[warn] icm targets are all -1; icm loss skipped.")
            if outputs["logits_te"] is not None and "te" in batch:
                te_targets = batch["te"].long()
                if (te_targets != -1).any():
                    loss_te = self.loss_te_fn(outputs["logits_te"], te_targets)
                    loss = loss + self.loss_w_te * loss_te
                    te_preds = torch.argmax(outputs["logits_te"], dim=1)
                    valid = te_targets != -1
                    self.te_correct += int((te_preds[valid] == te_targets[valid]).sum().item())
                    self.te_total += int(valid.sum().item())
                else:
                    if batch_idx == 0:
                        print("[warn] te targets are all -1; te loss skipped.")

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(float(loss), device=self.device)
        batch_size = batch["image"].shape[0]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_end(self):
        if self.val_probs:
            probs = torch.cat(self.val_probs)
            targets = torch.cat(self.val_targets)
            best_t, best_f1 = find_best_threshold(probs, targets)
            self.best_threshold = best_t

            val_acc = self.val_acc.compute()
            val_auroc = self.val_auroc.compute()
            val_auprc = self.val_auprc.compute()

            self.log("val_acc", val_acc, prog_bar=True)
            self.log("val_auroc", val_auroc, prog_bar=True)
            self.log("val_auprc", val_auprc, prog_bar=True)
            self.log("val_f1", torch.tensor(best_f1, device=self.device), prog_bar=True)
            self.log("val_best_threshold", torch.tensor(best_t, device=self.device))
        if self.stage_total > 0:
            self.log("val_stage_acc", self.stage_correct / self.stage_total, prog_bar=False)
        if self.morph_total > 0:
            self.log("val_morph_acc", self.morph_correct / self.morph_total, prog_bar=False)
        if self.exp_total > 0:
            self.log("val_exp_acc", self.exp_correct / self.exp_total, prog_bar=False)
        if self.icm_total > 0:
            self.log("val_icm_acc", self.icm_correct / self.icm_total, prog_bar=False)
        if self.te_total > 0:
            self.log("val_te_acc", self.te_correct / self.te_total, prog_bar=False)

        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_probs = []
        self.val_targets = []
        self.stage_correct = 0
        self.stage_total = 0
        self.morph_correct = 0
        self.morph_total = 0
        self.exp_correct = 0
        self.exp_total = 0
        self.icm_correct = 0
        self.icm_total = 0
        self.te_correct = 0
        self.te_total = 0

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_threshold"] = float(self.best_threshold)

    def on_load_checkpoint(self, checkpoint):
        if "best_threshold" in checkpoint:
            self.best_threshold = float(checkpoint["best_threshold"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.cfg.training.epochs),
            eta_min=float(self.cfg.training.lr) * 0.05,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def run_training(cfg, overfit_n=0):
    set_seed(cfg.seed)

    dm = HVDataModule(cfg, overfit_n=overfit_n)
    dm.setup()

    output_dir = Path(cfg.logging.output_dir)
    ensure_dir(output_dir)
    save_config(cfg, output_dir / "config.yaml")

    split_df = dm.get_split_df()
    if split_df is not None:
        split_df.to_csv(output_dir / "splits.csv", index=False)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tb")

    training_cfg = _cfg_get(cfg, "training", cfg)
    early_cfg = _cfg_get(training_cfg, "early_stopping", {})
    monitor = str(_cfg_get(early_cfg, "monitor", "val_auprc"))
    early_enabled = bool(_cfg_get(early_cfg, "enabled", True))
    patience = int(_cfg_get(early_cfg, "patience", 10))
    mode = str(_cfg_get(early_cfg, "mode", "auto")).lower()

    if dm.val_df is not None and len(dm.val_df) == 0 and monitor.startswith("val_"):
        monitor = "train_loss_epoch"
        print("[warn] val split empty; switching monitor to train_loss_epoch.")

    if mode == "auto":
        mode = "min" if "loss" in monitor else "max"

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=False,
        filename="best",
        enable_version_counter=False,
    )
    callbacks = [ckpt_cb]
    if early_enabled:
        es_cb = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
        )
        callbacks.append(es_cb)

    precision = int(cfg.training.precision)
    if precision == 16 and not torch.cuda.is_available():
        precision = 32

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.epochs),
        accelerator=str(cfg.training.accelerator),
        devices=int(cfg.training.devices),
        precision=precision,
        deterministic=bool(cfg.training.deterministic),
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
    )

    model = LitClassifier(cfg, pos_weight=dm.pos_weight)
    init_ckpt = _cfg_get(_cfg_get(cfg, "training", cfg), "init_ckpt", None)
    if init_ckpt:
        state = torch.load(init_ckpt, map_location="cpu")
        state_dict = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[init] missing keys: {len(missing)}")
            for key in missing[:10]:
                print(f"[init]   missing: {key}")
        if unexpected:
            print(f"[init] unexpected keys: {len(unexpected)}")
            for key in unexpected[:10]:
                print(f"[init]   unexpected: {key}")
        print("[init] loaded checkpoint weights (strict=False)")

    trainable = 0
    frozen = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    total = trainable + frozen
    if total > 0:
        print(f"[freeze] trainable params: {trainable} | frozen params: {frozen}")
    trainer.fit(model, dm)

    return output_dir, ckpt_cb.best_model_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_n", type=int, default=0)
    parser.add_argument("--init_ckpt", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.init_ckpt is not None:
        cfg.training.init_ckpt = args.init_ckpt

    output_dir, best_ckpt = run_training(cfg, overfit_n=args.overfit_n)
    print(f"Output dir: {output_dir}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
