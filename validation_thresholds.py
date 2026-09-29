"""Calibrate image-level thresholds on validation and report held-out results.

First generate validation predictions with:
    python predict.py --weights runs/yolo26s_640/weights/best.pt --split val --sweep

Then run:
    python validation_thresholds.py --target-sensitivity 0.95

The threshold is selected separately for each inference configuration using only
validation image scores. That frozen threshold is then applied to the held-out
test/dev image scores. The script does not use test labels to choose thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parent / "results"


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = {"image_name", "score", "truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if df["image_name"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate image names")
    scores = df["score"].to_numpy(dtype=float)
    truth = df["truth"].to_numpy(dtype=int)
    if not np.isfinite(scores).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError(f"{path.name} contains invalid scores or labels")
    return scores, truth


def image_metrics(scores: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float | int]:
    flagged = scores >= threshold
    positive = truth == 1
    clean = truth == 0
    tp = int(np.sum(flagged & positive))
    fp = int(np.sum(flagged & clean))
    n_positive = int(np.sum(positive))
    n_clean = int(np.sum(clean))
    n_flagged = tp + fp
    return {
        "n_images": int(len(truth)),
        "n_positive_images": n_positive,
        "n_clean_images": n_clean,
        "sensitivity": tp / n_positive if n_positive else float("nan"),
        "specificity": (n_clean - fp) / n_clean if n_clean else float("nan"),
        "ppv": tp / n_flagged if n_flagged else float("nan"),
        "flagged_per_100_clean": 100.0 * fp / n_clean if n_clean else float("nan"),
    }


def select_validation_threshold(
    scores: np.ndarray, truth: np.ndarray, target: float
) -> tuple[float | None, dict[str, float | int]]:
    if not 0 < target <= 1:
        raise ValueError("target sensitivity must be in (0, 1]")
    if not np.any(truth == 1):
        raise ValueError("validation predictions contain no positive images")

    # Match evaluate.py's threshold sweep, choosing the highest threshold that
    # still reaches the target sensitivity on validation data.
    thresholds = np.unique(np.concatenate((np.linspace(0.0, 1.0, 201), scores)))
    eligible = []
    for threshold in thresholds:
        metrics = image_metrics(scores, truth, float(threshold))
        if metrics["sensitivity"] >= target:
            eligible.append((float(threshold), metrics))
    if not eligible:
        return None, image_metrics(scores, truth, 0.0)
    return max(eligible, key=lambda item: item[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    results = args.results

    val_paths = sorted(results.glob("pred_cls_val_*.csv"))
    if not val_paths:
        raise SystemExit(
            f"No validation predictions found in {results}. Run predict.py with --split val --sweep first."
        )

    output_rows = []
    for val_path in val_paths:
        tag = val_path.stem.removeprefix("pred_cls_val_")
        test_path = results / f"pred_cls_test_{tag}.csv"
        if not test_path.exists():
            raise SystemExit(f"Missing matching held-out predictions: {test_path.name}")

        val_scores, val_truth = read_predictions(val_path)
        test_scores, test_truth = read_predictions(test_path)
        threshold, val_metrics = select_validation_threshold(
            val_scores, val_truth, args.target_sensitivity
        )
        row = {
            "tag": tag,
            "target_sensitivity": args.target_sensitivity,
            "threshold_selected_on": "validation",
            "threshold_achievable_on_validation": threshold is not None,
            "threshold_from_validation": threshold,
        }
        row.update({f"validation_{key}": value for key, value in val_metrics.items()})
        if threshold is None:
            row["validation_max_sensitivity"] = val_metrics["sensitivity"]
            for key in ("n_images", "n_positive_images", "n_clean_images", "sensitivity", "specificity", "ppv", "flagged_per_100_clean"):
                row[f"test_{key}_at_validation_threshold"] = np.nan
        else:
            test_metrics = image_metrics(test_scores, test_truth, threshold)
            row.update({f"test_{key}_at_validation_threshold": value for key, value in test_metrics.items()})
        output_rows.append(row)

    out_path = results / "operating_point_val_to_test.csv"
    pd.DataFrame(output_rows).sort_values("tag").to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(pd.DataFrame(output_rows)[[
        "tag", "threshold_from_validation", "validation_sensitivity",
        "test_sensitivity_at_validation_threshold",
        "test_specificity_at_validation_threshold",
        "test_flagged_per_100_clean_at_validation_threshold",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
