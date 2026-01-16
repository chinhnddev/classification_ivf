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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        logits = self(batch["image"])
        targets = batch["label"].float()
        loss = self.loss_fn(logits, targets)
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
        logits = self(batch["image"])
        targets = batch["label"].float()
        loss = self.loss_fn(logits, targets)
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

        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_probs = []
        self.val_targets = []

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
