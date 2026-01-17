import argparse
from pathlib import Path
import pandas as pd

from hv.utils import load_config


def validate(csv_path, data_root):
    df = pd.read_csv(csv_path)
    root = Path(data_root)

    if "label" not in df.columns and "quality" in df.columns:
        df["label"] = df["quality"]

    if "label" in df.columns and df["label"].dtype == object:
        mapping = {"good": 1, "poor": 0, "bad": 0, "1": 1, "0": 0}
        df["label"] = df["label"].astype(str).str.lower().map(mapping)

    missing = []
    for rel in df["image_path"].tolist():
        if not (root / str(rel)).is_file():
            missing.append(rel)

    label_counts = df["label"].value_counts().to_dict()
    day_counts = df["day"].value_counts().to_dict() if "day" in df.columns else None
    embryo_count = df["embryo_id"].nunique() if "embryo_id" in df.columns else 0

    print(f"Total samples: {len(df)}")
    print(f"Label counts: {label_counts}")
    if day_counts is not None:
        print(f"Day counts: {day_counts}")
    if "embryo_id" in df.columns:
        print(f"Unique embryo_id: {embryo_count}")
    print(f"Missing files: {len(missing)}")
    if missing:
        print("First missing paths:", missing[:5])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    csv_path = args.csv if args.csv is not None else cfg.data.csv_path
    validate(csv_path, cfg.data.data_root)


if __name__ == "__main__":
    main()
