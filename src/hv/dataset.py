from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl


def build_transforms(cfg, is_train, disable_augment=False):
    resize_size = int(cfg.data.resize_size)
    crop_size = int(cfg.data.crop_size)
    mean = cfg.data.normalize_mean
    std = cfg.data.normalize_std

    if is_train and not disable_augment:
        rotation = float(cfg.data.rotation_deg)
        jitter = float(cfg.data.color_jitter)
        hflip = float(cfg.data.horizontal_flip)
        t = [
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(crop_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        ]
        if rotation > 0:
            t.append(transforms.RandomRotation(degrees=rotation))
        if hflip > 0:
            t.append(transforms.RandomHorizontalFlip(p=hflip))
        if jitter > 0:
            t.append(transforms.ColorJitter(brightness=jitter, contrast=jitter))
        t.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        return transforms.Compose(t)

    t = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(t)


def _check_split_fracs(train_frac, val_frac, test_frac):
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")


def make_image_split(df, seed, train_frac, val_frac, test_frac):
    _check_split_fracs(train_frac, val_frac, test_frac)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_total = len(df)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def stratified_split(df, seed, train_frac, val_frac, test_frac):
    _check_split_fracs(train_frac, val_frac, test_frac)
    if df["label"].nunique() < 2:
        return make_image_split(df, seed, train_frac, val_frac, test_frac)

    rng = np.random.RandomState(seed)
    train_parts = []
    val_parts = []
    test_parts = []
    for label, group in df.groupby("label"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_total = len(idx)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        train_parts.append(df.loc[idx[:n_train]])
        val_parts.append(df.loc[idx[n_train:n_train + n_val]])
        test_parts.append(df.loc[idx[n_train + n_val:]])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def normalize_labels(df):
    if "label" not in df.columns:
        if "quality" in df.columns:
            df = df.copy()
            df["label"] = df["quality"]
        else:
            raise KeyError("Missing label column. Expected 'label' or 'quality'.")

    if df["label"].dtype == object:
        mapping = {
            "good": 1,
            "poor": 0,
            "bad": 0,
            "1": 1,
            "0": 0,
        }
        df = df.copy()
        df["label"] = df["label"].astype(str).str.lower().map(mapping)
    df["label"] = df["label"].astype(int)
    return df


MISSING_TOKENS = {"", "0", "ND", "NA", "N/A"}
GARDNER_RE = re.compile(r"^([1-6])([ABC])([ABC])$")


def _normalize_token(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        token = value.strip().upper()
        if token in MISSING_TOKENS:
            return None
        return token
    if isinstance(value, (int, float)):
        if float(value) == 0.0:
            return None
        return value
    return value


def _is_missing(value):
    return _normalize_token(value) is None


def _is_range_token(value):
    token = _normalize_token(value)
    return isinstance(token, str) and "-" in token


def _parse_exp(value):
    value = _normalize_token(value)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    exp = int(value)
    if 1 <= exp <= 6:
        return exp
    return None


def _parse_icm_te(value):
    value = _normalize_token(value)
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().upper()
        if token in {"A", "B", "C"}:
            return token
        try:
            value = float(token)
        except ValueError:
            return None
    idx = int(value)
    if 1 <= idx <= 3:
        return ["A", "B", "C"][idx - 1]
    return None


def _parse_gardner(value):
    value = _normalize_token(value)
    if value is None or not isinstance(value, str):
        return None
    token = value.strip().upper()
    if "-" in token:
        return None
    match = GARDNER_RE.match(token)
    if not match:
        return None
    exp = int(match.group(1))
    icm = match.group(2)
    te = match.group(3)
    return exp, icm, te


def prepare_morphology_df(df):
    exp_ids = []
    icm_ids = []
    te_ids = []
    keep_indices = []

    for idx, row in df.iterrows():
        if _is_range_token(row.get("gardner")) or _is_range_token(row.get("grade")):
            continue

        exp = _parse_exp(row.get("exp"))
        icm = _parse_icm_te(row.get("icm"))
        te = _parse_icm_te(row.get("te"))

        if exp is None or icm is None or te is None:
            gardner_val = row.get("gardner")
            if _is_missing(gardner_val):
                gardner_val = row.get("grade")
            parsed = _parse_gardner(gardner_val)
            if parsed:
                parsed_exp, parsed_icm, parsed_te = parsed
                if exp is None:
                    exp = parsed_exp
                if icm is None:
                    icm = parsed_icm
                if te is None:
                    te = parsed_te

        if exp is None:
            continue
        if exp < 3:
            exp_ids.append(exp - 1)
            icm_ids.append(-1)
            te_ids.append(-1)
            keep_indices.append(idx)
            continue
        if icm is None or te is None:
            continue

        exp_ids.append(exp - 1)
        icm_ids.append({"A": 0, "B": 1, "C": 2}[icm])
        te_ids.append({"A": 0, "B": 1, "C": 2}[te])
        keep_indices.append(idx)

    if not keep_indices:
        return df.iloc[0:0].copy()

    out = df.loc[keep_indices].copy().reset_index(drop=True)
    out["exp"] = exp_ids
    out["icm"] = icm_ids
    out["te"] = te_ids
    return out


def prepare_stage_df(df):
    mapping = {"cleavage": 0, "morula": 1, "blastocyst": 2}
    stage_ids = []
    keep_indices = []
    for idx, value in df["stage"].items():
        token = _normalize_token(value)
        if token is None:
            continue
        key = token.strip().lower()
        if key not in mapping:
            continue
        stage_ids.append(mapping[key])
        keep_indices.append(idx)

    if not keep_indices:
        return df.iloc[0:0].copy()

    out = df.loc[keep_indices].copy().reset_index(drop=True)
    out["stage"] = stage_ids
    return out


def compute_pos_weight(df):
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    if pos == 0:
        return 1.0
    return float(neg / pos)


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    image_paths = [b["image_path"] for b in batch]
    days = torch.stack([b["day"] for b in batch])
    output = {"image": images, "label": labels, "image_path": image_paths, "day": days}
    if "stage" in batch[0]:
        output["stage"] = torch.stack([b["stage"] for b in batch])
    if "morph" in batch[0]:
        output["morph"] = torch.stack([b["morph"] for b in batch])
    if "exp" in batch[0]:
        output["exp"] = torch.stack([b["exp"] for b in batch])
    if "icm" in batch[0]:
        output["icm"] = torch.stack([b["icm"] for b in batch])
    if "te" in batch[0]:
        output["te"] = torch.stack([b["te"] for b in batch])
    return output


class HVEmbryoDataset(Dataset):
    def __init__(self, df, data_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform
        self.has_stage = "stage" in self.df.columns
        self.has_morph = "morph" in self.df.columns
        self.has_exp = "exp" in self.df.columns
        self.has_icm = "icm" in self.df.columns
        self.has_te = "te" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.data_root / str(row["image_path"])
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if "label" in row.index:
            label_value = float(row["label"])
        else:
            label_value = -1.0
        label = torch.tensor(label_value, dtype=torch.float32)
        if "day" in row.index:
            day_raw = row["day"]
            if pd.isna(day_raw):
                day_value = -1
            else:
                day_value = int(day_raw)
        else:
            day_value = -1
        day = torch.tensor(day_value, dtype=torch.long)
        sample = {
            "image": img,
            "label": label,
            "image_path": str(row["image_path"]),
            "day": day,
        }
        if self.has_stage:
            stage_val = row.get("stage")
            if pd.isna(stage_val):
                stage_val = -1
            sample["stage"] = torch.tensor(int(stage_val), dtype=torch.long)
        if self.has_morph:
            morph_val = row.get("morph")
            if pd.isna(morph_val):
                morph_val = -1
            sample["morph"] = torch.tensor(int(morph_val), dtype=torch.long)
        if self.has_exp:
            exp_val = row.get("exp")
            if pd.isna(exp_val):
                exp_val = -1
            sample["exp"] = torch.tensor(int(exp_val), dtype=torch.long)
        if self.has_icm:
            icm_val = row.get("icm")
            if pd.isna(icm_val):
                icm_val = -1
            sample["icm"] = torch.tensor(int(icm_val), dtype=torch.long)
        if self.has_te:
            te_val = row.get("te")
            if pd.isna(te_val):
                te_val = -1
            sample["te"] = torch.tensor(int(te_val), dtype=torch.long)
        return sample


class HVDataModule(pl.LightningDataModule):
    def __init__(self, cfg, overfit_n=0):
        super().__init__()
        self.cfg = cfg
        self.seed = int(cfg.seed)
        self.csv_path = cfg.data.csv_path
        self.data_root = cfg.data.data_root
        self.overfit_n = int(overfit_n)
        self.disable_augment = self.overfit_n > 0
        self.eval_external = bool(getattr(cfg.data, "eval_external", False))
        self.use_predefined_split = bool(getattr(cfg.data, "use_predefined_split", False))
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.pos_weight = 1.0

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        task = str(getattr(self.cfg.data, "task", "quality")).lower()
        if task in {"quality", "all"}:
            df = normalize_labels(df)

        if task == "morphology":
            df = prepare_morphology_df(df)
        elif task == "stage":
            if "stage" not in df.columns:
                raise KeyError("Missing 'stage' column for stage task.")
            df = prepare_stage_df(df)

        if self.eval_external:
            self.test_df = df.reset_index(drop=True)
            self.train_df = pd.DataFrame(columns=df.columns)
            self.val_df = pd.DataFrame(columns=df.columns)
        elif self.overfit_n > 0:
            n = min(self.overfit_n, len(df))
            df = df.sample(n=n, random_state=self.seed).reset_index(drop=True)
            self.train_df = df.copy()
            self.val_df = df.copy()
            self.test_df = df.copy()
        elif self.use_predefined_split:
            if "split" not in df.columns:
                raise KeyError("Expected 'split' column for predefined split.")
            self.train_df = df[df["split"] == "train"].reset_index(drop=True)
            self.val_df = df[df["split"] == "val"].reset_index(drop=True)
            self.test_df = df[df["split"] == "test"].reset_index(drop=True)
        else:
            self.train_df, self.val_df, self.test_df = stratified_split(
                df,
                self.seed,
                self.cfg.data.train_frac,
                self.cfg.data.val_frac,
                self.cfg.data.test_frac,
            )

        if self.eval_external:
            if "label" in self.test_df.columns:
                self.pos_weight = compute_pos_weight(self.test_df)
            else:
                self.pos_weight = 1.0
        else:
            if "label" in self.train_df.columns:
                self.pos_weight = compute_pos_weight(self.train_df)
            else:
                self.pos_weight = 1.0

        train_transform = build_transforms(
            self.cfg,
            is_train=not self.eval_external,
            disable_augment=self.disable_augment or self.eval_external,
        )
        eval_transform = build_transforms(self.cfg, is_train=False)

        self.train_ds = HVEmbryoDataset(self.train_df, self.data_root, transform=train_transform)
        self.val_ds = HVEmbryoDataset(self.val_df, self.data_root, transform=eval_transform)
        self.test_ds = HVEmbryoDataset(
            self.test_df,
            self.data_root,
            transform=eval_transform,
        )
        self._log_dataset_summary()

    def _log_dataset_summary(self):
        if self.eval_external:
            print(f"[data] external size: {len(self.test_df)}")
            if "label" in self.test_df.columns:
                counts = self.test_df["label"].value_counts().to_dict()
                good = int(counts.get(1, 0))
                poor = int(counts.get(0, 0))
                print(f"[data] external good/poor: {good}/{poor}")
            return

        def _counts(frame):
            if "label" not in frame.columns:
                return None, None
            counts = frame["label"].value_counts().to_dict()
            good = int(counts.get(1, 0))
            poor = int(counts.get(0, 0))
            return good, poor

        train_good, train_poor = _counts(self.train_df)
        val_good, val_poor = _counts(self.val_df)
        test_good, test_poor = _counts(self.test_df)
        print(
            "[data] train/val/test sizes: "
            f"{len(self.train_df)}/{len(self.val_df)}/{len(self.test_df)}"
        )
        if train_good is not None:
            print(
                "[data] train good/poor: "
                f"{train_good}/{train_poor} | val good/poor: {val_good}/{val_poor} | "
                f"test good/poor: {test_good}/{test_poor}"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

    def get_split_df(self):
        if self.train_df is None:
            return None
        train = self.train_df.copy()
        train["split"] = "train"
        val = self.val_df.copy()
        val["split"] = "val"
        test = self.test_df.copy()
        test["split"] = "test"
        return pd.concat([train, val, test], axis=0).reset_index(drop=True)
