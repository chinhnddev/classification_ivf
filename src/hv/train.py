import argparse
from pathlib import Path
import torch
import torch.nn as nn
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


class LitClassifier(pl.LightningModule):
    def __init__(self, cfg, pos_weight=1.0):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.train_acc = BinaryAccuracy(threshold=0.5)
        self.val_acc = BinaryAccuracy(threshold=0.5)
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.val_probs = []
        self.val_targets = []
        self.best_threshold = 0.5
        model_cfg = _cfg_get(cfg, "model", cfg)
        self.enable_stage = bool(_cfg_get(model_cfg, "enable_stage", False))
        self.enable_morph = bool(_cfg_get(model_cfg, "enable_morph", False))
        weights_cfg = _cfg_get(_cfg_get(cfg, "training", cfg), "loss_weights", None)
        self.loss_w_quality = float(_cfg_get(weights_cfg, "quality", 1.0))
        self.loss_w_stage = float(_cfg_get(weights_cfg, "stage", 0.3))
        self.loss_w_morph = float(_cfg_get(weights_cfg, "morph", 0.3))
        self.loss_stage_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_morph_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.stage_correct = 0
        self.stage_total = 0
        self.morph_correct = 0
        self.morph_total = 0

    def forward(self, x):
        return self.model(x)

    def _unpack_outputs(self, outputs):
        if isinstance(outputs, dict):
            return outputs
        return {"logits_quality": outputs, "logits_stage": None, "logits_morph": None}

    def training_step(self, batch, batch_idx):
        outputs = self(batch["image"])
        outputs = self._unpack_outputs(outputs)
        logits = outputs["logits_quality"]
        targets = batch["label"].float()
        loss_quality = self.loss_fn(logits, targets)
        loss = self.loss_w_quality * loss_quality
        if self.enable_stage and outputs["logits_stage"] is not None and "stage" in batch:
            stage_targets = batch["stage"].long()
            if (stage_targets != -1).any():
                loss_stage = self.loss_stage_fn(outputs["logits_stage"], stage_targets)
                loss = loss + self.loss_w_stage * loss_stage
        if self.enable_morph and outputs["logits_morph"] is not None and "morph" in batch:
            morph_targets = batch["morph"].long()
            if (morph_targets != -1).any():
                loss_morph = self.loss_morph_fn(outputs["logits_morph"], morph_targets)
                loss = loss + self.loss_w_morph * loss_morph
        probs = torch.sigmoid(logits)
        self.train_acc.update(probs, targets.int())
        batch_size = targets.shape[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
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
        targets = batch["label"].float()
        loss_quality = self.loss_fn(logits, targets)
        loss = self.loss_w_quality * loss_quality
        if self.enable_stage and outputs["logits_stage"] is not None and "stage" in batch:
            stage_targets = batch["stage"].long()
            if (stage_targets != -1).any():
                loss_stage = self.loss_stage_fn(outputs["logits_stage"], stage_targets)
                loss = loss + self.loss_w_stage * loss_stage
                stage_preds = torch.argmax(outputs["logits_stage"], dim=1)
                valid = stage_targets != -1
                self.stage_correct += int((stage_preds[valid] == stage_targets[valid]).sum().item())
                self.stage_total += int(valid.sum().item())
        if self.enable_morph and outputs["logits_morph"] is not None and "morph" in batch:
            morph_targets = batch["morph"].long()
            if (morph_targets != -1).any():
                loss_morph = self.loss_morph_fn(outputs["logits_morph"], morph_targets)
                loss = loss + self.loss_w_morph * loss_morph
                morph_preds = torch.argmax(outputs["logits_morph"], dim=1)
                valid = morph_targets != -1
                self.morph_correct += int((morph_preds[valid] == morph_targets[valid]).sum().item())
                self.morph_total += int(valid.sum().item())
        probs = torch.sigmoid(logits)
        self.val_acc.update(probs, targets.int())
        self.val_auroc.update(probs, targets.int())
        self.val_auprc.update(probs, targets.int())
        self.val_probs.append(probs.detach().cpu())
        self.val_targets.append(targets.detach().cpu())
        batch_size = targets.shape[0]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_probs:
            return
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

        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_probs = []
        self.val_targets = []
        self.stage_correct = 0
        self.stage_total = 0
        self.morph_correct = 0
        self.morph_total = 0

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

    monitor = str(cfg.training.early_stopping.monitor)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        monitor=monitor,
        mode="max",
        save_top_k=1,
        save_last=False,
        filename="best",
        enable_version_counter=False,
    )
    es_cb = EarlyStopping(
        monitor=monitor,
        mode="max",
        patience=int(cfg.training.early_stopping.patience),
    )

    precision = int(cfg.training.precision)
    if precision == 16 and not torch.cuda.is_available():
        precision = 32

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.epochs),
        accelerator=str(cfg.training.accelerator),
        devices=int(cfg.training.devices),
        precision=precision,
        deterministic=bool(cfg.training.deterministic),
        callbacks=[ckpt_cb, es_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
    )

    model = LitClassifier(cfg, pos_weight=dm.pos_weight)
    trainer.fit(model, dm)

    return output_dir, ckpt_cb.best_model_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_n", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = int(args.seed)

    output_dir, best_ckpt = run_training(cfg, overfit_n=args.overfit_n)
    print(f"Output dir: {output_dir}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
