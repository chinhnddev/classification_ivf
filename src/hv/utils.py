from pathlib import Path
import random
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path):
    return OmegaConf.load(path)


def save_config(cfg, path):
    ensure_dir(Path(path).parent)
    OmegaConf.save(config=cfg, f=str(path))
