"""
Build the YOLO-format dataset from raw object-CXR.

    python prepare_data.py

What it does
------------
1. Verifies the raw download is complete and matches the published counts.
2. Splits the 8,000 official training images into 7,000 train / 1,000 val,
   stratified on whether the image contains a foreign object.
3. Converts annotations to YOLO bounding-box labels (training geometry only).
4. Symlinks images into the layout Ultralytics expects and writes the data yaml.
5. Re-derives a random sample of labels from the source CSV and asserts they
   match what was written.

Split discipline
----------------
The official 1,000-image test split was never publicly released. `dev` is
therefore held out as the test set and is NEVER used for tuning. Validation is
carved from train. Tuning on dev and then reporting dev is the one mistake in
this project that cannot be repaired after the fact.

Output layout
-------------
    $DS/
      train/images/*.jpg -> symlinks   train/labels/*.txt
      val/images/*.jpg   -> symlinks   val/labels/*.txt
      test/images/*.jpg  -> symlinks   test/labels/*.txt      (= official dev)
      object-cxr.yaml
      split_manifest.csv     which image went where, for the paper's appendix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config as C
import ocxr

EXPECTED = {"train": 8000, "dev": 1000}
EXPECTED_POSITIVES = {"train": 4000, "dev": 500}


# --------------------------------------------------------------------------


def verify_raw() -> None:
    """Fail loudly and specifically if the download is missing or partial."""
    problems: list[str] = []

    for p in (C.RAW_TRAIN_CSV, C.RAW_DEV_CSV):
        if not p.exists():
            problems.append(f"missing {p}")
    for d in (C.RAW_TRAIN_DIR, C.RAW_DEV_DIR):
        if not d.is_dir():
            problems.append(f"missing directory {d}")

    if problems:
        raise SystemExit(
            "Raw dataset not ready:\n  "
            + "\n  ".join(problems)
            + "\n\nRun the Kaggle download first (see README, step 2)."
        )

    for name, d in (("train", C.RAW_TRAIN_DIR), ("dev", C.RAW_DEV_DIR)):
        n = len(list(d.glob("*.jpg")))
        if n != EXPECTED[name]:
            raise SystemExit(
                f"{d} holds {n} jpgs, expected {EXPECTED[name]}. "
                "The unzip was probably interrupted — delete the folder and re-download."
            )


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) read from the JPEG header only — no pixel decoding."""
    with Image.open(path) as im:
        return im.size


def stratified_split(df: pd.DataFrame, n_val: int, seed: int) -> pd.Series:
    """Assign each row 'train' or 'val', preserving the positive/negative ratio.

    object-CXR is exactly balanced (4,000 / 4,000). Splitting without
    stratification would leave that to chance and make the validation curve
    noisier than it needs to be.
    """
    rng = np.random.default_rng(seed)
    assignment = pd.Series("train", index=df.index, dtype=object)

    frac_val = n_val / len(df)
    for label in (0, 1):
        idx = df.index[df.has_object == label].to_numpy()
        rng.shuffle(idx)
        take = int(round(len(idx) * frac_val))
        assignment.loc[idx[:take]] = "val"

    # Rounding can leave us a row or two off; nudge the majority class.
    while (assignment == "val").sum() > n_val:
        cand = assignment.index[assignment == "val"][-1]
        assignment.loc[cand] = "train"
    while (assignment == "val").sum() < n_val:
        cand = assignment.index[assignment == "train"][-1]
        assignment.loc[cand] = "val"

    return assignment


def build_split(
    name: str, rows: pd.DataFrame, src_dir: Path, root: Path
) -> dict[str, int]:
    """Materialise one split: symlinked images plus YOLO label files."""
    img_dir, lbl_dir = root / name / "images", root / name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_boxes = n_dropped = n_empty = 0

    for r in rows.itertuples(index=False):
        src = src_dir / r.image_name
        dst = img_dir / r.image_name
        if not dst.exists():
            dst.symlink_to(src)

        w, h = image_size(src)
        lines = ocxr.to_yolo_lines(r.objects, w, h)
        n_dropped += len(r.objects) - len(lines)
        n_boxes += len(lines)
        if not lines:
            n_empty += 1

        # A background image gets an empty (not absent) label file. Ultralytics
        # treats those as explicit negatives, which is exactly what we want:
        # half this dataset is deliberately object-free.
        (lbl_dir / f"{Path(r.image_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    return {
        "images": len(rows),
        "boxes": n_boxes,
        "empty_label_files": n_empty,
        "objects_dropped": n_dropped,
    }


def verify_written(root: Path, split: str, rows: pd.DataFrame, src_dir: Path, n: int, seed: int) -> None:
    """Re-derive n random labels from source and assert the files match.

    Catches silent corruption: wrong image paired with wrong label, coordinate
    transposition, stale files from an earlier run.
    """
    rng = np.random.default_rng(seed)
    positives = rows[rows.n_objects > 0]
    if positives.empty:
        return
    sample = positives.iloc[rng.choice(len(positives), size=min(n, len(positives)), replace=False)]

    for r in sample.itertuples(index=False):
        w, h = image_size(src_dir / r.image_name)
        expected = ocxr.to_yolo_lines(r.objects, w, h)
        actual = (root / split / "labels" / f"{Path(r.image_name).stem}.txt").read_text().split("\n")
        actual = [ln for ln in actual if ln.strip()]
        if expected != actual:
            raise AssertionError(
                f"label mismatch for {r.image_name}\n  expected {expected}\n  actual   {actual}"
            )


def write_yaml(root: Path) -> None:
    names = "\n".join(f"  {k}: {v}" for k, v in C.CLASS_NAMES.items())
    C.DATA_YAML.write_text(
        f"# Generated by prepare_data.py — do not edit by hand.\n"
        f"path: {root}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: {len(C.CLASS_NAMES)}\n"
        f"names:\n{names}\n"
    )


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild even if $DS already exists")
    args = ap.parse_args()

    C.seed_everything()
    print(C.describe(), "\n")

    verify_raw()

    if C.DS.exists() and not args.force:
        print(f"{C.DS} already exists. Re-run with --force to rebuild.")
        return 0

    print("loading annotations ...")
    train_df = ocxr.load_split(C.RAW_TRAIN_CSV)
    dev_df = ocxr.load_split(C.RAW_DEV_CSV)

    for name, df in (("train", train_df), ("dev", dev_df)):
        pos = int(df.has_object.sum())
        if pos != EXPECTED_POSITIVES[name]:
            raise SystemExit(
                f"{name}.csv has {pos} positives, expected {EXPECTED_POSITIVES[name]}. "
                "This is not the dataset this pipeline was written for."
            )
        print(f"  {name}: {len(df)} images, {pos} positive, {ocxr.type_counts(df)}")

    print("\nsplitting train -> train/val (stratified) ...")
    train_df["split"] = stratified_split(train_df, C.N_VAL, C.SEED)
    for s in ("train", "val"):
        sub = train_df[train_df.split == s]
        print(f"  {s}: {len(sub)} images, {int(sub.has_object.sum())} positive")

    print("\nbuilding ...")
    stats = {
        "train": build_split("train", train_df[train_df.split == "train"], C.RAW_TRAIN_DIR, C.DS),
        "val": build_split("val", train_df[train_df.split == "val"], C.RAW_TRAIN_DIR, C.DS),
        "test": build_split("test", dev_df, C.RAW_DEV_DIR, C.DS),
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nverifying a sample of written labels against source ...")
    verify_written(C.DS, "train", train_df[train_df.split == "train"], C.RAW_TRAIN_DIR, 25, C.SEED)
    verify_written(C.DS, "test", dev_df, C.RAW_DEV_DIR, 25, C.SEED)
    print("  ok")

    write_yaml(C.DS)

    manifest = pd.concat(
        [
            train_df[["image_name", "has_object", "n_objects", "split"]],
            dev_df[["image_name", "has_object", "n_objects"]].assign(split="test"),
        ]
    )
    manifest.to_csv(C.DS / "split_manifest.csv", index=False)
    manifest.to_csv(C.RESULTS / "split_manifest.csv", index=False)

    (C.RESULTS / "prepare_data_stats.json").write_text(
        json.dumps(
            {
                "seed": C.SEED,
                "splits": stats,
                "annotation_types_train": ocxr.type_counts(train_df),
                "annotation_types_dev": ocxr.type_counts(dev_df),
            },
            indent=2,
        )
    )

    print(f"\ndone. data yaml at {C.DATA_YAML}")
    print(f"split manifest copied to {C.RESULTS / 'split_manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
