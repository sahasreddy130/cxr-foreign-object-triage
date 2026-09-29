"""
Score predictions: FROC, AUC, operating points, per-category sensitivity.

    python evaluate.py --split test                 # every prediction file found
    python evaluate.py --split test --tag whole640  # just one
    python evaluate.py --split test --verify-froc   # cross-check against froc.py

Outputs into $RESULTS:

    metrics_<split>.csv         one row per configuration — the paper's main table
    froc_curve_<tag>.csv        sensitivity vs false-positives-per-image
    roc_<tag>.csv               image-level ROC points
    operating_points_<tag>.csv  threshold sweep for the deployment analysis
    per_category_<tag>.csv      sensitivity by object category (needs Phase 4)

Two kinds of false positive
---------------------------
Both are reported and they answer different questions.

  Localisation FP  a predicted point falling outside every annotation. This is
                   what FROC counts, and it measures how noisy the detector is.

  Image-level FP   a foreign-object-free radiograph that gets flagged. This is
                   what a re-filming triage system actually costs a hospital,
                   because it is a film someone has to look at again.

The headline deployment number is image-level: at the threshold achieving 95%
sensitivity, how many clean films out of 100 are flagged unnecessarily?
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import ocxr


# --------------------------------------------------------------------------
# FROC
# --------------------------------------------------------------------------


def froc_curve(
    gt: pd.DataFrame, preds: dict[str, list[tuple[float, float, float]]]
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sensitivity as a function of false positives per image.

    Faithful to the challenge's froc.py: predictions from every image are pooled
    and ranked by confidence; a prediction is a hit if it falls inside an
    as-yet-unhit annotation on its own image, and a false positive otherwise. An
    object already found by a higher-confidence prediction cannot be found twice.

    Returns (fp_per_image, sensitivity, n_objects, n_images), both arrays in
    prediction order so they can be plotted directly.
    """
    by_image = {r.image_name: r.objects for r in gt.itertuples(index=False)}
    n_images = len(gt)
    n_objects = int(gt.n_objects.sum())
    if n_objects == 0:
        raise ValueError("ground truth contains no annotated objects")

    pooled: list[tuple[float, str, float, float]] = []
    for name, pts in preds.items():
        for prob, x, y in pts:
            pooled.append((prob, name, x, y))
    pooled.sort(key=lambda t: -t[0])

    hit_objects: set[tuple[str, int]] = set()
    hits = fps = 0
    fp_axis, sens_axis = [], []

    for prob, name, x, y in pooled:
        matched = False
        for obj in by_image.get(name, []):
            if obj.contains(x, y):
                matched = True
                key = (name, obj.index)
                if key not in hit_objects:
                    hit_objects.add(key)
                    hits += 1
        if not matched:
            fps += 1
        fp_axis.append(fps / n_images)
        sens_axis.append(hits / n_objects)

    return np.asarray(fp_axis), np.asarray(sens_axis), n_objects, n_images


def froc_score(fp: np.ndarray, sens: np.ndarray, fps_points: list[float]) -> tuple[float, list[float]]:
    """Average sensitivity at the official false-positive rates.

    At each rate, sensitivity is read at the first prediction where the pooled
    false-positive rate reaches that rate.

    Carry-forward semantics — subtle, and it changes the number
    ----------------------------------------------------------
    If the model never produces enough false positives to reach a rate, the
    reference implementation carries forward the sensitivity recorded at the last
    rate that *was* reached — not the final sensitivity after all predictions are
    consumed. Those differ whenever the tail of the ranked list contains hits.

    Using the final sensitivity instead inflates FROC by several points. We match
    the reference exactly, because the whole purpose of reporting FROC here is
    comparability with the published leaderboard.

    Practical consequence: a model that emits too few low-confidence detections
    is silently penalised, because it cannot reach the permissive end of the
    curve at all. Keep PRED_CONF at 0.001 and PRED_MAX_DET generous.
    """
    if len(sens) == 0:
        return 0.0, [0.0] * len(fps_points)

    out: list[float] = []
    for rate in fps_points:
        idx = int(np.searchsorted(fp, rate, side="left"))
        if idx >= len(fp):
            out.append(out[-1] if out else float(sens[-1]))
        else:
            out.append(float(sens[idx]))
    return float(np.mean(out)), out


def max_fp_rate(fp: np.ndarray) -> float:
    """Highest false-positive rate the predictions actually reach.

    If this is below max(FROC_FPS), the tail of the FROC curve is extrapolated
    rather than measured, and the score is not directly comparable to a model
    that does reach it. Report it.
    """
    return float(fp[-1]) if len(fp) else 0.0


def official_froc(gt_csv: Path, pred_csv: Path) -> float | None:
    """Run the challenge's froc.py and parse its FROC.

    A cross-check, not the primary path. If our reimplementation and the
    reference disagree by more than rounding, something is wrong and you want to
    know before it reaches the paper.
    """
    script = Path(__file__).with_name("froc.py")
    if not script.exists():
        return None
    try:
        out = subprocess.run(
            [sys.executable, str(script), str(gt_csv), str(pred_csv)],
            capture_output=True, text=True, timeout=900, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    (official froc.py failed: {e})")
        return None
    m = re.search(r"FROC:\s*\n\s*([0-9.]+)", out)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------
# Image-level classification
# --------------------------------------------------------------------------


def roc_and_auc(truth: np.ndarray, score: np.ndarray) -> tuple[pd.DataFrame, float]:
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, thr = roc_curve(truth, score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}), float(roc_auc_score(truth, score))


def operating_points(
    truth: np.ndarray,
    score: np.ndarray,
    preds: dict[str, list[tuple[float, float, float]]],
    names: list[str],
) -> pd.DataFrame:
    """Sweep the decision threshold; report what each choice costs.

    `flagged_per_100_clean` is the number a hospital cares about: of 100 films
    with nothing on them, how many does the system send back for re-filming?
    """
    thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), score]))
    n_pos, n_neg = int(truth.sum()), int((1 - truth).sum())
    rows = []

    for t in thresholds:
        flagged = score >= t
        tp = int(((truth == 1) & flagged).sum())
        fp = int(((truth == 0) & flagged).sum())
        detections = sum(sum(1 for p, _, _ in preds.get(n, []) if p >= t) for n in names)
        rows.append(
            {
                "threshold": float(t),
                "sensitivity": tp / n_pos if n_pos else np.nan,
                "specificity": (n_neg - fp) / n_neg if n_neg else np.nan,
                "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
                "flagged_per_100_clean": 100.0 * fp / n_neg if n_neg else np.nan,
                # This is the number of predicted points per image, not the
                # number of points outside the ground-truth regions.
                "detections_per_image": detections / len(names),
            }
        )
    return pd.DataFrame(rows)


def at_target_sensitivity(ops: pd.DataFrame, target: float) -> dict:
    """Highest threshold that still achieves the target sensitivity."""
    ok = ops[ops.sensitivity >= target]
    if ok.empty:
        return {"achievable": False, "max_sensitivity": float(ops.sensitivity.max())}
    row = ok.loc[ok.threshold.idxmax()]
    return {
        "achievable": True,
        "threshold": float(row.threshold),
        "sensitivity": float(row.sensitivity),
        "specificity": float(row.specificity),
        "ppv": float(row.ppv),
        "flagged_per_100_clean": float(row.flagged_per_100_clean),
        "detections_per_image": float(row.detections_per_image),
    }


# --------------------------------------------------------------------------
# Per-category (requires Phase 4 annotations)
# --------------------------------------------------------------------------


def per_category(
    gt: pd.DataFrame, preds: dict[str, list[tuple[float, float, float]]], threshold: float
) -> pd.DataFrame | None:
    """Sensitivity by object category at a fixed operating point.

    Needs $ANNOTATIONS/categories_final.csv with columns image_name,
    object_index, category — produced by annotate.py --merge.

    This is the table the original draft claimed without the labels to support
    it. Reported here it is earned: the categories are yours, the agreement
    statistic is published alongside, and the mapping to objects is explicit.
    """
    path = C.ANNOTATIONS / "categories_final.csv"
    if not path.exists():
        return None

    cats = pd.read_csv(path)
    required = {"image_name", "object_index", "category"}
    if not required.issubset(cats.columns):
        raise SystemExit(f"{path} needs columns {sorted(required)}")

    by_image = {r.image_name: r.objects for r in gt.itertuples(index=False)}
    lookup = {(r.image_name, int(r.object_index)): r.category for r in cats.itertuples(index=False)}

    found: dict[tuple[str, int], bool] = {}
    for name, objs in by_image.items():
        pts = [(p, x, y) for p, x, y in preds.get(name, []) if p >= threshold]
        for obj in objs:
            found[(name, obj.index)] = any(obj.contains(x, y) for _, x, y in pts)

    rows = []
    for key, hit in found.items():
        cat = lookup.get(key)
        if cat is None:
            continue  # object outside the labelled subset
        rows.append({"category": cat, "hit": int(hit)})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    out = (
        df.groupby("category")
        .agg(n_objects=("hit", "size"), n_detected=("hit", "sum"))
        .reset_index()
    )
    out["sensitivity"] = out.n_detected / out.n_objects
    # Wilson interval — with per-category counts in the tens, a bare proportion
    # invites over-reading. Report the interval in the paper.
    z = 1.96
    p, n = out.sensitivity, out.n_objects
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    out["ci_low"] = (centre - half).clip(0, 1)
    out["ci_high"] = (centre + half).clip(0, 1)
    return out.sort_values("sensitivity", ascending=False)


def validation_threshold_for(tag: str) -> float | None:
    """Return a configuration's frozen validation threshold, when available."""
    path = C.RESULTS / "operating_point_val_to_test.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    rows = table.loc[table.tag == tag, "threshold_from_validation"]
    if rows.empty or pd.isna(rows.iloc[0]):
        return None
    return float(rows.iloc[0])


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tag", default=None, help="evaluate only this configuration")
    ap.add_argument("--verify-froc", action="store_true", help="cross-check against the official froc.py")
    ap.add_argument("--target-sensitivity", type=float, default=C.TARGET_SENSITIVITY)
    args = ap.parse_args()

    gt_csv = C.RAW_DEV_CSV if args.split == "test" else C.RAW_TRAIN_CSV
    gt = ocxr.load_split(gt_csv)
    if args.split in ("train", "val"):
        manifest_path = C.RESULTS / "split_manifest.csv"
        if not manifest_path.exists():
            raise SystemExit(
                f"split manifest not found at {manifest_path}; run prepare_data.py first"
            )
        manifest = pd.read_csv(manifest_path)
        names = set(manifest.loc[manifest.split == args.split, "image_name"])
        gt = gt[gt.image_name.isin(names)].reset_index(drop=True)
        if gt.empty:
            raise SystemExit(f"split manifest contains no ground truth rows for split={args.split}")

    pattern = f"pred_loc_{args.split}_" + (args.tag if args.tag else "*") + ".csv"
    files = sorted(C.RESULTS.glob(pattern))
    if not files:
        raise SystemExit(f"no prediction files matching {pattern} in {C.RESULTS}")

    print(f"ground truth: {len(gt)} images, {int(gt.n_objects.sum())} objects")
    print(f"reference:    AUC {C.LEADERBOARD_AUC}  FROC {C.LEADERBOARD_FROC} (JF Healthcare baseline)\n")

    summary = []

    for loc_path in files:
        tag = loc_path.stem.replace(f"pred_loc_{args.split}_", "")
        print(f"[{tag}]")

        preds = ocxr.read_localization_csv(loc_path)

        # A filename mismatch is the classic silent failure: FROC comes out near
        # zero and looks like a modelling result.
        overlap = set(preds) & set(gt.image_name)
        if len(overlap) < 0.99 * len(gt):
            print(
                f"    !! only {len(overlap)}/{len(gt)} prediction filenames match the ground truth.\n"
                f"       Expected bare names like '{gt.image_name.iloc[0]}'; "
                f"got '{next(iter(preds))}'."
            )

        fp_axis, sens_axis, n_obj, n_img = froc_curve(gt, preds)
        froc, sens_at = froc_score(fp_axis, sens_axis, C.FROC_FPS)

        reached = max_fp_rate(fp_axis)
        if reached < max(C.FROC_FPS):
            print(
                f"    !! predictions only reach {reached:.2f} FP/image, "
                f"below the {max(C.FROC_FPS)} the metric asks for.\n"
                f"       The tail of the curve is carried forward, not measured, and FROC is\n"
                f"       understated. Lower PRED_CONF or raise PRED_MAX_DET and re-predict."
            )

        pd.DataFrame({"fp_per_image": fp_axis, "sensitivity": sens_axis}).to_csv(
            C.RESULTS / f"froc_curve_{args.split}_{tag}.csv", index=False
        )

        official = None
        if args.verify_froc:
            official = official_froc(gt_csv, loc_path)
            if official is not None:
                delta = abs(official - froc)
                mark = "ok" if delta < 5e-3 else "MISMATCH"
                print(f"    froc.py cross-check: {official:.4f} vs {froc:.4f}  [{mark}]")

        cls_path = C.RESULTS / f"pred_cls_{args.split}_{tag}.csv"
        auc = np.nan
        op_target: dict = {}
        if cls_path.exists():
            cls = pd.read_csv(cls_path)
            roc, auc = roc_and_auc(cls.truth.values, cls.score.values)
            roc.to_csv(C.RESULTS / f"roc_{args.split}_{tag}.csv", index=False)

            ops = operating_points(cls.truth.values, cls.score.values, preds, list(cls.image_name))
            ops.to_csv(C.RESULTS / f"operating_points_{args.split}_{tag}.csv", index=False)
            op_target = at_target_sensitivity(ops, args.target_sensitivity)

            category_threshold = op_target.get("threshold", 0.25)
            if args.split == "test":
                category_threshold = validation_threshold_for(tag)
                if category_threshold is None:
                    print("    per-category results skipped: run validation threshold calibration first")
            if category_threshold is not None:
                cat = per_category(gt, preds, category_threshold)
                if cat is not None:
                    cat.to_csv(C.RESULTS / f"per_category_{args.split}_{tag}.csv", index=False)
                    print(
                        f"    per-category sensitivity written at threshold "
                        f"{category_threshold:.4f} ({len(cat)} categories)"
                    )

        print(f"    FROC {froc:.4f}   AUC {auc:.4f}")
        print(f"    sensitivity at {C.FROC_FPS} FP/img:")
        print("      " + "  ".join(f"{s:.3f}" for s in sens_at))
        if op_target.get("achievable"):
            print(
                f"    at {args.target_sensitivity:.0%} sensitivity: threshold {op_target['threshold']:.3f}, "
                f"{op_target['flagged_per_100_clean']:.1f} clean films flagged per 100"
            )
        elif op_target:
            print(
                f"    {args.target_sensitivity:.0%} sensitivity unreachable "
                f"(max {op_target['max_sensitivity']:.3f})"
            )
        print()

        selected_prefix = "test_selected_" if args.split == "test" else "validation_selected_"
        op_columns = {
            (
                f"{selected_prefix}threshold_achievable"
                if key == "achievable"
                else f"{selected_prefix}{key}"
            ): value
            for key, value in op_target.items()
        }
        summary.append(
            {
                "tag": tag,
                "froc": round(froc, 4),
                "auc": round(float(auc), 4),
                "official_froc": official,
                "n_objects": n_obj,
                "n_images": n_img,
                **{f"sens_at_{r}": round(s, 4) for r, s in zip(C.FROC_FPS, sens_at)},
                **op_columns,
            }
        )

    out = pd.DataFrame(summary).sort_values("froc", ascending=False)
    path = C.RESULTS / f"metrics_{args.split}.csv"
    out.to_csv(path, index=False)

    print("=" * 72)
    print(out[["tag", "froc", "auc"]].to_string(index=False))
    print("=" * 72)
    print(f"\nwritten to {path}")
    print("Next: python figures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
