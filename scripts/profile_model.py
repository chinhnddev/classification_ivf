import argparse
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from hv.models import build_model
from hv.utils import load_config


def profile_model(cfg_path, input_sizes):
    cfg = load_config(cfg_path)
    model = build_model(cfg)
    model.eval()
    for size in input_sizes:
        inputs = torch.zeros(1, 3, size, size)
        with FlopCounterMode(model, display=False) as fc:
            _ = model(inputs)
        flops = fc.get_total_flops()
        params = sum(p.numel() for p in model.parameters())
        print(f"input={size} params={params} flops={flops}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_sizes", type=int, nargs="*", default=[224, 192])
    return parser.parse_args()


def main():
    args = parse_args()
    profile_model(args.config, args.input_sizes)


if __name__ == "__main__":
    main()
