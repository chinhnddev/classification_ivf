import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_RE = re.compile(r"ED[1-4]", re.IGNORECASE)
CATEGORY_RE = re.compile(r"(\d+)")


def iter_images(root):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def infer_source(path):
    for part in path.parts:
        match = SOURCE_RE.search(part)
        if match:
            return match.group(0).upper()
    return None


def infer_category_id(path):
    last_dir = path.parent.name
    if last_dir in {"1", "2", "3", "4", "5"}:
        return int(last_dir)
    match = CATEGORY_RE.search(last_dir)
    if match:
        return int(match.group(1))
    return None


def map_label(source, category_id):
    if source in {"ED1", "ED2", "ED4"}:
        if category_id in {1, 2}:
            return 0
        if category_id in {3, 4, 5}:
            return 1
        return None
    if source == "ED3":
        if category_id == 1:
            return 0
        if category_id == 2:
            return 1
        return None
    return None


def _split_indices(idx, test_frac):
    n_total = len(idx)
    n_test = int(math.ceil(n_total * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def stratified_train_test_split(df, test_frac, seed):
    rng = np.random.RandomState(seed)
    if df["label"].nunique() < 2:
        idx = df.index.to_numpy()
        rng.shuffle(idx)
        train_idx, test_idx = _split_indices(idx, test_frac)
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)
        return train_df, test_df

    train_parts = []
    test_parts = []
    for _, group in df.groupby("label"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        train_idx, test_idx = _split_indices(idx, test_frac)
        train_parts.append(df.loc[train_idx])
        test_parts.append(df.loc[test_idx])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df


def _counts(df):
    counts = df["label"].value_counts().to_dict()
    good = int(counts.get(1, 0))
    bad = int(counts.get(0, 0))
    return good, bad


def build_metadata(data_root, dataset_dir, seed, test_frac, val_frac_within_train):
    data_root = Path(data_root).resolve()
    dataset_dir = Path(dataset_dir).resolve()

    rows = []
    skipped_unlabeled = []
    skipped_invalid_source = []

    for image_path in iter_images(dataset_dir):
        rel_path = Path(os.path.relpath(image_path.resolve(), data_root)).as_posix()
        source = infer_source(image_path)
        if source is None:
            skipped_invalid_source.append(rel_path)
            continue

        category_id = infer_category_id(image_path)
        if category_id is None:
            skipped_unlabeled.append(rel_path)
            continue

        label = map_label(source, category_id)
        if label is None:
            skipped_unlabeled.append(rel_path)
            continue

        rows.append(
            {
                "image_path": rel_path,
                "label": int(label),
                "source": source,
                "category_id": int(category_id),
            }
        )

    df = pd.DataFrame(rows, columns=["image_path", "label", "source", "category_id"])
    if df.empty:
        raise SystemExit("No labeled images found. Check dataset_dir and folder layout.")

    train_pool_df, test_df = stratified_train_test_split(df, test_frac, seed)
    train_df, val_df = stratified_train_test_split(train_pool_df, val_frac_within_train, seed)
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    full_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    return full_df, train_df, val_df, test_df, skipped_unlabeled, skipped_invalid_source


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--val_frac_within_train", type=float, default=0.2)
    return parser.parse_args()


def resolve_output_path(path):
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_v{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    args = parse_args()
    out_csv = resolve_output_path(Path(args.out_csv))
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_path = out_dir / "skipped_unlabeled.txt"
    invalid_source_path = out_dir / "skipped_invalid_source.txt"

    (
        full_df,
        train_df,
        val_df,
        test_df,
        skipped_unlabeled,
        skipped_invalid_source,
    ) = build_metadata(
        args.data_root,
        args.dataset_dir,
        args.seed,
        args.test_frac,
        args.val_frac_within_train,
    )

    full_df.to_csv(out_csv, index=False)
    if out_csv != Path(args.out_csv):
        print(f"[output] existing csv found, wrote new file: {out_csv}")

    unlabeled_path.write_text("\n".join(skipped_unlabeled), encoding="utf-8")
    invalid_source_path.write_text("\n".join(skipped_invalid_source), encoding="utf-8")

    train_good, train_bad = _counts(train_df)
    val_good, val_bad = _counts(val_df)
    test_good, test_bad = _counts(test_df)
    print(f"[split] total size: {len(full_df)}")
    print(
        "[split] train/val/test sizes: "
        f"{len(train_df)}/{len(val_df)}/{len(test_df)}"
    )
    print(
        "[split] train good/bad: "
        f"{train_good}/{train_bad} | val good/bad: {val_good}/{val_bad} | "
        f"test good/bad: {test_good}/{test_bad}"
    )
    print(f"[skip] unlabeled: {len(skipped_unlabeled)}")
    print(f"[skip] invalid source: {len(skipped_invalid_source)}")


if __name__ == "__main__":
    main()
