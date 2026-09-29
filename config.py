"""
Central configuration for the object-CXR foreign-object triage study.

Every path, constant and hyperparameter used anywhere in the pipeline is defined
here. Nothing else in the codebase should hard-code a path.

Reproducibility contract
------------------------
SEED fixes the train/val split, the annotator assignment in annotate.py, and the
torch/numpy RNGs in train.py. Changing it changes results; record it in the paper.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

SEED = 1337


def seed_everything(seed: int = SEED) -> None:
    """Seed every RNG we can reach. Call at the top of any script that samples."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN costs a little speed and buys reproducible training.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Environment detection
# --------------------------------------------------------------------------

IN_COLAB = Path("/content").exists()

# Scratch space. Large, fast, and wiped when the VM is recycled.
SCRATCH = Path("/content") if IN_COLAB else Path.cwd() / "scratch"

# Persistent space. Small files only — never put the dataset here.
if IN_COLAB:
    DRIVE = Path("/content/drive/MyDrive/CXR-Foreign-Objects")
else:
    DRIVE = Path.cwd() / "workdir"


# --------------------------------------------------------------------------
# Dataset layout
# --------------------------------------------------------------------------

KAGGLE_SLUG = "raddar/foreign-objects-in-chest-xrays"

# Where the Kaggle archive unpacks to.
RAW = SCRATCH / "kag" / "object-CXR"
RAW_TRAIN_DIR = RAW / "train"          # 8,000 jpgs
RAW_DEV_DIR = RAW / "dev"              # 1,000 jpgs
RAW_TRAIN_CSV = RAW / "train.csv"
RAW_DEV_CSV = RAW / "dev.csv"

# Where prepare_data.py builds the YOLO-format dataset.
DS = SCRATCH / "ds"
DATA_YAML = DS / "object-cxr.yaml"

# Split sizes. The official 1,000-image test split was never publicly released,
# so `dev` is held out as our test set and validation is carved from train.
N_VAL = 1000                            # taken from the 8,000 train images
CLASS_NAMES = {0: "foreign object"}

# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

RESULTS = DRIVE / "results"
FIGURES = DRIVE / "figures"
ANNOTATIONS = DRIVE / "annotations"
WEIGHTS = DRIVE / "weights"
RUNS = DRIVE / "runs"

for _d in (RESULTS, FIGURES, ANNOTATIONS, WEIGHTS):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Model and training
# --------------------------------------------------------------------------

# YOLO26 (Ultralytics, January 2026). Scales: n, s, m, l, x.
# 's' is the sweet spot for a free Colab T4: trains in a few hours, clearly
# stronger than 'n'. Move to 'm' only if you have compute to spare.
MODEL = "yolo26s.pt"

EPOCHS = 100
BATCH = 64
TRAIN_IMGSZ = 640
PATIENCE = 25          # early stopping; 0 disables

# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

# FROC sweeps the operating point itself, so predictions must be collected at a
# near-zero confidence floor. Filtering at the usual 0.25 truncates the curve and
# silently understates FROC. Do not raise this.
PRED_CONF = 0.001
PRED_MAX_DET = 300

# Resolutions for the core experiment.
IMGSZ_SWEEP = [640, 960, 1280]

# Sliced ("tiled") inference, after Akyon et al. 2022.
TILE_SIZE = 640
TILE_OVERLAP = 0.2      # fraction of tile size
# Two predicted points closer than this (in original-image pixels) are merged,
# keeping the higher confidence. Prevents overlap regions inflating false
# positives. Roughly a third of a tile.
TILE_MERGE_DIST = 200

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

# The official FROC operating points (JF Healthcare, object-CXR challenge).
FROC_FPS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

# Target sensitivity for the deployment-oriented operating point analysis.
TARGET_SENSITIVITY = 0.95

# Published reference numbers, for context in plots and tables.
LEADERBOARD_AUC = 0.9209
LEADERBOARD_FROC = 0.8031

# --------------------------------------------------------------------------
# Phase 4 annotation
# --------------------------------------------------------------------------

CATEGORIES = [
    "button_zipper",
    "jewellery_chain",
    "coin",
    "clip_pin",
    "medical_hardware",
    "other",
    "unclear",
]

N_ANNOTATORS = 4
OVERLAP_N = 50          # images every annotator labels, for inter-rater agreement
CROP_PAD = 0.35         # context padding around each object crop, as a fraction


def describe() -> str:
    """Human-readable dump of the active configuration. Print this into logs."""
    lines = [
        "object-CXR foreign-object triage — configuration",
        f"  seed            {SEED}",
        f"  environment     {'Colab' if IN_COLAB else 'local'}",
        f"  scratch         {SCRATCH}",
        f"  persistent      {DRIVE}",
        f"  model           {MODEL}",
        f"  epochs / batch  {EPOCHS} / {BATCH}",
        f"  train imgsz     {TRAIN_IMGSZ}",
        f"  imgsz sweep     {IMGSZ_SWEEP}",
        f"  tile / overlap  {TILE_SIZE} / {TILE_OVERLAP}",
        f"  pred conf floor {PRED_CONF}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
