"""
Generate every figure in the paper.

    python figures.py                 # all of them
    python figures.py --only froc roc

Writes 300-dpi PNGs to $FIGURES, one per figure, named fig1_... through fig6_...

Figures
-------
    fig1_annotation_types     the three annotation geometries on real radiographs
    fig2_froc                 FROC curves, one per configuration
    fig3_roc                  image-level ROC
    fig4_operating_point      sensitivity and false-flag cost vs threshold
    fig5_per_category         sensitivity by object category, with 95% CIs
    fig6_qualitative          successes and failures side by side

House style
-----------
Colour-blind-safe palette, and every series is also distinguishable by line
style, because journals still print in greyscale and judges still print
handouts. No chart junk, no gridline soup, axis labels with units.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

import config as C
import ocxr

Image.MAX_IMAGE_PIXELS = None

# Okabe-Ito, colour-blind safe.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)


def _save(fig, name: str) -> None:
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def _pretty(tag: str) -> str:
    if tag.startswith("whole"):
        return f"Whole image, {tag.replace('whole', '')} px"
    if tag.startswith("tiled"):
        parts = tag.replace("tiled", "").split("_ov")
        base = f"Tiled {parts[0]} px"
        if len(parts) > 1:
            base += f", {parts[1].replace('_sahi', '')}% overlap"
        return base + (" (SAHI)" if tag.endswith("sahi") else "")
    return tag


# --------------------------------------------------------------------------


def fig_annotation_types(split_csv: Path, img_dir: Path) -> None:
    """One example of each annotation geometry, drawn on the radiograph."""
    df = ocxr.load_split(split_csv)
    wanted = {ocxr.RECTANGLE: None, ocxr.ELLIPSE: None, ocxr.POLYGON: None}

    for r in df.itertuples(index=False):
        for obj in r.objects:
            if wanted.get(obj.type) is None:
                wanted[obj.type] = (r.image_name, obj)
        if all(v is not None for v in wanted.values()):
            break

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    for ax, (atype, payload) in zip(axes, wanted.items()):
        if payload is None:
            ax.axis("off")
            continue
        name, obj = payload
        with Image.open(img_dir / name) as im:
            im = im.convert("L")
            x1, y1, x2, y2 = obj.to_bbox()
            pad = max(220.0, 0.7 * max(x2 - x1, y2 - y1))
            box = (
                int(max(0, x1 - pad)), int(max(0, y1 - pad)),
                int(min(im.size[0], x2 + pad)), int(min(im.size[1], y2 + pad)),
            )
            ax.imshow(np.asarray(im.crop(box)), cmap="gray")

        ox, oy = box[0], box[1]
        if obj.type == ocxr.RECTANGLE:
            ax.add_patch(mpatches.Rectangle((x1 - ox, y1 - oy), x2 - x1, y2 - y1,
                                            fill=False, ec=PALETTE[1], lw=1.8))
        elif obj.type == ocxr.ELLIPSE:
            ax.add_patch(mpatches.Ellipse(((x1 + x2) / 2 - ox, (y1 + y2) / 2 - oy),
                                          x2 - x1, y2 - y1, fill=False, ec=PALETTE[1], lw=1.8))
        else:
            pts = obj.coords.reshape(-1, 2) - np.array([ox, oy])
            ax.add_patch(mpatches.Polygon(pts, closed=True, fill=False, ec=PALETTE[1], lw=1.8))
            ax.add_patch(mpatches.Rectangle((x1 - ox, y1 - oy), x2 - x1, y2 - y1,
                                            fill=False, ec=PALETTE[0], lw=1.0, ls="--"))

        ax.set_title(f"{ocxr.TYPE_NAMES[atype].capitalize()}\n{name}")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    fig.suptitle("Annotation geometries in object-CXR", y=1.02)
    fig.text(
        0.5, -0.04,
        "Orange: the annotation as recorded. Blue dashed: the enclosing box used for training.\n"
        "For polygons the two differ substantially, which is why evaluation uses FROC on the "
        "true shape rather than IoU on the box.",
        ha="center", fontsize=8.5,
    )
    _save(fig, "fig1_annotation_types")


def fig_froc(split: str) -> None:
    files = sorted(C.RESULTS.glob(f"froc_curve_{split}_*.csv"))
    if not files:
        print("  (no FROC curves found — run evaluate.py)")
        return

    metrics = C.RESULTS / f"metrics_{split}.csv"
    scores = {}
    if metrics.exists():
        m = pd.read_csv(metrics)
        scores = dict(zip(m.tag, m.froc))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, f in enumerate(files):
        tag = f.stem.replace(f"froc_curve_{split}_", "")
        d = pd.read_csv(f)
        lbl = _pretty(tag) + (f"  (FROC {scores[tag]:.3f})" if tag in scores else "")
        ax.plot(d.fp_per_image, d.sensitivity, color=PALETTE[i % len(PALETTE)],
                ls=STYLES[i % len(STYLES)], lw=1.8, label=lbl)

    for r in C.FROC_FPS:
        ax.axvline(r, color="0.85", lw=0.6, zorder=0)
    ax.axhline(C.LEADERBOARD_FROC, color="0.4", lw=1.0, ls=(0, (1, 2)))
    ax.text(8.2, C.LEADERBOARD_FROC, f" challenge baseline\n FROC {C.LEADERBOARD_FROC}",
            va="center", fontsize=8, color="0.35")

    ax.set_xscale("log")
    ax.set_xticks(C.FROC_FPS)
    ax.set_xticklabels([str(r) for r in C.FROC_FPS])
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Free-response ROC by inference configuration")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "fig2_froc")


def fig_roc(split: str) -> None:
    files = sorted(C.RESULTS.glob(f"roc_{split}_*.csv"))
    if not files:
        print("  (no ROC files found — run evaluate.py)")
        return

    metrics = C.RESULTS / f"metrics_{split}.csv"
    aucs = {}
    if metrics.exists():
        m = pd.read_csv(metrics)
        aucs = dict(zip(m.tag, m.auc))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for i, f in enumerate(files):
        tag = f.stem.replace(f"roc_{split}_", "")
        d = pd.read_csv(f)
        lbl = _pretty(tag) + (f"  (AUC {aucs[tag]:.3f})" if tag in aucs else "")
        ax.plot(d.fpr, d.tpr, color=PALETTE[i % len(PALETTE)],
                ls=STYLES[i % len(STYLES)], lw=1.8, label=lbl)

    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Image-level detection of any foreign object")
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal")
    _save(fig, "fig3_roc")


def fig_operating_point(split: str, tag: str | None) -> None:
    files = sorted(C.RESULTS.glob(f"operating_points_{split}_*.csv"))
    if not files:
        print("  (no operating-point files — run evaluate.py)")
        return
    f = next((x for x in files if tag and tag in x.stem), files[0])
    chosen = f.stem.replace(f"operating_points_{split}_", "")
    d = pd.read_csv(f).sort_values("threshold")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(d.threshold, d.sensitivity, color=PALETTE[0], lw=1.9,
            label="Held-out sensitivity")
    ax.plot(d.threshold, d.specificity, color=PALETTE[2], lw=1.6, ls="--",
            label="Held-out specificity")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.02)

    ax2 = ax.twinx()
    ax2.plot(d.threshold, d.flagged_per_100_clean, color=PALETTE[1], lw=1.6, ls="-.",
             label="Held-out clean films flagged per 100")
    ax2.set_ylabel("Clean films flagged per 100")
    ax2.grid(False)

    calibration_path = C.RESULTS / "operating_point_val_to_test.csv"
    if calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        selected = calibration.loc[calibration.tag == chosen]
        if not selected.empty:
            row = selected.iloc[0]
            t = float(row.threshold_from_validation)
            test_sensitivity = float(row.test_sensitivity_at_validation_threshold)
            flagged = float(row.test_flagged_per_100_clean_at_validation_threshold)
            ax.axvline(t, color="0.35", lw=1.0, ls=(0, (1, 2)),
                       label="Validation-selected threshold")
            ax.scatter([t], [test_sensitivity], color=PALETTE[0], zorder=4)
            ax2.scatter([t], [flagged], color=PALETTE[1], marker="D", zorder=4)
            subtitle = (
                f"Validation-selected threshold {t:.3f}; held-out sensitivity "
                f"{test_sensitivity:.1%}\n{flagged:.1f} clean films flagged per 100"
            )
        else:
            subtitle = "Validation-calibrated operating point unavailable"
    else:
        print("  (no validation-calibrated operating point — run validation_thresholds.py)")
        subtitle = "Validation-calibrated operating point unavailable"

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.27)
    fig.legend(
        h1 + h2,
        l1 + l2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    ax.set_title(
        f"Held-out operating-point trade-off — {_pretty(chosen)}\n{subtitle}",
        fontsize=9.5,
        pad=9,
    )
    _save(fig, "fig4_operating_point")


def fig_per_category(split: str, tag: str | None) -> None:
    files = sorted(C.RESULTS.glob(f"per_category_{split}_*.csv"))
    if not files:
        print("  (no per-category file — complete Phase 4 and re-run evaluate.py)")
        return
    f = next((x for x in files if tag and tag in x.stem), files[0])
    d = pd.read_csv(f).sort_values("sensitivity")

    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(d) + 1.9))
    y = np.arange(len(d))
    err = np.vstack([d.sensitivity - d.ci_low, d.ci_high - d.sensitivity])
    ax.barh(y, d.sensitivity, color=PALETTE[0], alpha=0.85, height=0.62)
    ax.errorbar(d.sensitivity, y, xerr=err, fmt="none", ecolor="0.25", capsize=3, lw=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in d.category])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Sensitivity")
    ax.set_title("Detection sensitivity by object category")
    for i, r in enumerate(d.itertuples(index=False)):
        ax.text(0.012, i, f"n = {r.n_objects}", va="center", fontsize=8, color="white"
                if r.sensitivity > 0.18 else "0.25")
    fig.text(0.5, -0.03,
             "Error bars are 95% Wilson intervals. Categories are author-assigned; "
             "see the reported Fleiss' kappa.", ha="center", fontsize=8.5)
    _save(fig, "fig5_per_category")


def fig_qualitative(split_csv: Path, img_dir: Path, split: str, tag: str | None, n: int = 3) -> None:
    """Detected and missed objects side by side, at a fixed operating point."""
    files = sorted(C.RESULTS.glob(f"pred_loc_{split}_*.csv"))
    if not files:
        print("  (no predictions — run predict.py)")
        return
    f = next((x for x in files if tag and tag in x.stem), files[0])
    preds = ocxr.read_localization_csv(f)

    thr = 0.25
    chosen = f.stem.replace(f"pred_loc_{split}_", "")
    calibration_path = C.RESULTS / "operating_point_val_to_test.csv"
    if split == "test" and calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        selected = calibration.loc[calibration.tag == chosen, "threshold_from_validation"]
        if not selected.empty and pd.notna(selected.iloc[0]):
            thr = float(selected.iloc[0])

    df = ocxr.load_split(split_csv)
    hits, misses = [], []
    for r in df[df.n_objects > 0].itertuples(index=False):
        pts = [(p, x, y) for p, x, y in preds.get(r.image_name, []) if p >= thr]
        for obj in r.objects:
            (hits if any(obj.contains(x, y) for _, x, y in pts) else misses).append((r.image_name, obj))
        if len(hits) >= n and len(misses) >= n:
            break

    rows = [("Detected", hits[:n]), ("Missed", misses[:n])]
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 7.0))
    axes = np.atleast_2d(axes)

    for r_i, (label, items) in enumerate(rows):
        for c_i in range(n):
            ax = axes[r_i, c_i]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if c_i >= len(items):
                ax.axis("off")
                continue
            name, obj = items[c_i]
            x1, y1, x2, y2 = obj.to_bbox()
            pad = max(200.0, 0.8 * max(x2 - x1, y2 - y1))
            with Image.open(img_dir / name) as im:
                im = im.convert("L")
                box = (int(max(0, x1 - pad)), int(max(0, y1 - pad)),
                       int(min(im.size[0], x2 + pad)), int(min(im.size[1], y2 + pad)))
                ax.imshow(np.asarray(im.crop(box)), cmap="gray")
            ax.add_patch(mpatches.Rectangle(
                (x1 - box[0], y1 - box[1]), x2 - x1, y2 - y1,
                fill=False, ec=PALETTE[2] if r_i == 0 else PALETTE[1], lw=1.8))
            ax.set_title(f"{label} — {name}\n{int(obj.area()):,} px²", fontsize=8.5)

    fig.suptitle(f"Qualitative examples at confidence ≥ {thr:.2f}", y=0.995)
    _save(fig, "fig6_qualitative")


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tag", default=None, help="configuration to feature in single-config figures")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["annotations", "froc", "roc", "operating", "category", "qualitative"])
    args = ap.parse_args()

    want = set(args.only) if args.only else {
        "annotations", "froc", "roc", "operating", "category", "qualitative"
    }
    csv = C.RAW_DEV_CSV if args.split == "test" else C.RAW_TRAIN_CSV
    img_dir = C.RAW_DEV_DIR if args.split == "test" else C.RAW_TRAIN_DIR

    print(f"figures -> {C.FIGURES}")
    if "annotations" in want:
        fig_annotation_types(csv, img_dir)
    if "froc" in want:
        fig_froc(args.split)
    if "roc" in want:
        fig_roc(args.split)
    if "operating" in want:
        fig_operating_point(args.split, args.tag)
    if "category" in want:
        fig_per_category(args.split, args.tag)
    if "qualitative" in want:
        fig_qualitative(csv, img_dir, args.split, args.tag)

    print("\nManuscript reminders:")
    print("  Follow the target venue's figure caption and attribution style.")
    print("  Refer to every figure in the body text and cite reused or adapted images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
