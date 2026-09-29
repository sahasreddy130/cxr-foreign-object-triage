"""
Run inference and emit FROC-format prediction files.

    # whole-image inference at three resolutions
    python predict.py --weights W --split test --imgsz 640
    python predict.py --weights W --split test --imgsz 960
    python predict.py --weights W --split test --imgsz 1280

    # sliced inference at native resolution
    python predict.py --weights W --split test --tiled

    # the whole sweep in one go
    python predict.py --weights W --split test --sweep

Outputs, per configuration, into $RESULTS:

    pred_loc_<tag>.csv    image_name,prediction  (probability x y; ...)
    pred_cls_<tag>.csv    image_name,score,truth
    predict_<tag>.json    configuration and timing

Why the confidence floor is 0.001
---------------------------------
FROC sweeps the operating point itself, from strict to permissive, and reports
sensitivity at 0.125 ... 8 false positives per image. Filtering predictions at
the usual 0.25 removes the permissive end of that curve, and the resulting FROC
is not merely noisy but systematically too low. Collect everything; let the
metric do the thresholding.

Sliced inference
----------------
object-CXR radiographs run around 1760x2140. Resizing to 640x640 is a 3-4x
downsample, and a coin or a clip survives as a handful of pixels. Slicing runs
the detector over overlapping native-resolution tiles instead, after Akyon et
al. (2022), who report +5.3 to +6.8 AP from slicing alone across three detectors.

This is a self-contained implementation rather than a call into the `sahi`
package. Three reasons: FROC needs points, not boxes, so the box-merging logic
SAHI spends most of its complexity on is unnecessary here; a local
implementation has no version-coupling to a third-party adapter; and the merge
rule stays visible and auditable, which matters when it affects the headline
number. `--backend sahi` is available if you want to cross-check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config as C
import ocxr

Image.MAX_IMAGE_PIXELS = None  # radiographs are large; the decompression-bomb guard is not useful here


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------


def tile_origins(size: int, tile: int, overlap: float) -> list[int]:
    """Left (or top) coordinates of tiles covering `size` pixels.

    The final tile is pushed flush against the edge rather than padded, so every
    pixel is covered exactly once at minimum and the border is never sampled at
    a different effective resolution from the interior.
    """
    if size <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, size - tile + 1, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def merge_points(
    points: list[tuple[float, float, float]], min_dist: float
) -> list[tuple[float, float, float]]:
    """Collapse near-duplicate detections from overlapping tiles.

    Greedy, highest confidence first: keep a point, suppress everything within
    `min_dist` of it, repeat. Without this, an object sitting in an overlap
    region is detected twice and the second detection is scored as a false
    positive, penalising slicing for its own overlap.
    """
    if not points:
        return []
    ordered = sorted(points, key=lambda p: -p[0])
    kept: list[tuple[float, float, float]] = []
    for prob, x, y in ordered:
        if all((x - kx) ** 2 + (y - ky) ** 2 >= min_dist**2 for _, kx, ky in kept):
            kept.append((prob, x, y))
    return kept


# --------------------------------------------------------------------------
# Inference backends
# --------------------------------------------------------------------------


def predict_whole(model, path: Path, imgsz: int) -> list[tuple[float, float, float]]:
    """Whole-image inference. Returns [(prob, x, y), ...] in original pixels."""
    r = model.predict(
        str(path), imgsz=imgsz, conf=C.PRED_CONF, max_det=C.PRED_MAX_DET, verbose=False
    )[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    # xywh is already rescaled to the original image by Ultralytics.
    xywh = r.boxes.xywh.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    return [(float(c), float(b[0]), float(b[1])) for b, c in zip(xywh, conf)]


def predict_tiled(
    model,
    path: Path,
    tile: int,
    overlap: float,
    merge_dist: float,
    include_full: bool,
    batch: int = 16,
) -> list[tuple[float, float, float]]:
    """Sliced inference at native resolution, plus an optional whole-image pass.

    The whole-image pass is kept by default (as SAHI does) because large objects
    can exceed a single tile and are more reliably found at full field of view.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        W, H = im.size
        arr = np.asarray(im)

    xs = tile_origins(W, tile, overlap)
    ys = tile_origins(H, tile, overlap)

    crops: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for y0 in ys:
        for x0 in xs:
            crops.append(arr[y0 : y0 + tile, x0 : x0 + tile])
            offsets.append((x0, y0))

    points: list[tuple[float, float, float]] = []
    for i in range(0, len(crops), batch):
        chunk, offs = crops[i : i + batch], offsets[i : i + batch]
        results = model.predict(
            chunk, imgsz=tile, conf=C.PRED_CONF, max_det=C.PRED_MAX_DET, verbose=False
        )
        for res, (x0, y0) in zip(results, offs):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xywh = res.boxes.xywh.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            for b, c in zip(xywh, conf):
                points.append((float(c), float(b[0]) + x0, float(b[1]) + y0))

    if include_full:
        points.extend(predict_whole(model, path, tile))

    return merge_points(points, merge_dist)


def predict_tiled_sahi(model_path: Path, path: Path, tile: int, overlap: float):
    """Optional cross-check against the reference SAHI implementation."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    det = None
    for mtype in ("ultralytics", "yolov8"):  # adapter was renamed; try both
        try:
            det = AutoDetectionModel.from_pretrained(
                model_type=mtype, model_path=str(model_path), confidence_threshold=C.PRED_CONF
            )
            break
        except Exception:
            continue
    if det is None:
        raise SystemExit("could not construct a SAHI detection model for these weights")

    res = get_sliced_prediction(
        str(path),
        det,
        slice_height=tile,
        slice_width=tile,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=0,
    )
    out = []
    for p in res.object_prediction_list:
        x1, y1, x2, y2 = p.bbox.to_xyxy()
        out.append((float(p.score.value), (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return out


# --------------------------------------------------------------------------


def run_one(model, model_path: Path, df: pd.DataFrame, img_dir: Path, cfg: dict) -> tuple[list, list, float]:
    rows: list[tuple[str, list[tuple[float, float, float]]]] = []
    scores: list[float] = []
    t0 = time.time()

    for i, name in enumerate(df.image_name, 1):
        p = img_dir / name
        if cfg["tiled"]:
            if cfg["backend"] == "sahi":
                pts = predict_tiled_sahi(model_path, p, cfg["tile"], cfg["overlap"])
            else:
                pts = predict_tiled(
                    model, p, cfg["tile"], cfg["overlap"], cfg["merge_dist"], cfg["include_full"]
                )
        else:
            pts = predict_whole(model, p, cfg["imgsz"])

        rows.append((name, pts))
        scores.append(max((p_ for p_, _, _ in pts), default=0.0))

        if i % 100 == 0 or i == len(df):
            rate = i / max(1e-9, time.time() - t0)
            print(f"    {i}/{len(df)}  {rate:.1f} img/s", flush=True)

    return rows, scores, time.time() - t0


def tag_for(cfg: dict) -> str:
    if cfg["tiled"]:
        return f"tiled{cfg['tile']}_ov{int(cfg['overlap'] * 100)}" + ("_sahi" if cfg["backend"] == "sahi" else "")
    return f"whole{cfg['imgsz']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=C.TRAIN_IMGSZ)
    ap.add_argument("--tiled", action="store_true", help="sliced inference at native resolution")
    ap.add_argument("--tile", type=int, default=C.TILE_SIZE)
    ap.add_argument("--overlap", type=float, default=C.TILE_OVERLAP)
    ap.add_argument("--merge-dist", type=float, default=C.TILE_MERGE_DIST)
    ap.add_argument("--no-full-pass", action="store_true", help="tiles only, skip the whole-image pass")
    ap.add_argument("--backend", default="local", choices=["local", "sahi"])
    ap.add_argument("--sweep", action="store_true", help="all resolutions in IMGSZ_SWEEP, then tiled")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    args = ap.parse_args()

    C.seed_everything()

    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    csv = C.RAW_DEV_CSV if args.split == "test" else C.RAW_TRAIN_CSV
    df = ocxr.load_split(csv)

    # For train/val, restrict to the rows the split manifest assigned.
    if args.split in ("train", "val"):
        man = pd.read_csv(C.DS / "split_manifest.csv")
        keep = set(man.loc[man.split == args.split, "image_name"])
        df = df[df.image_name.isin(keep)].reset_index(drop=True)

    img_dir = C.RAW_DEV_DIR if args.split == "test" else C.RAW_TRAIN_DIR
    if args.limit:
        df = df.head(args.limit)

    print(f"{args.split}: {len(df)} images, {int(df.has_object.sum())} positive")
    if args.split == "test":
        print("(test == the official dev split; never tune on these numbers)")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    configs: list[dict] = []
    if args.sweep:
        for s in C.IMGSZ_SWEEP:
            configs.append(dict(tiled=False, imgsz=s))
        configs.append(dict(tiled=True, imgsz=args.tile))
    else:
        configs.append(dict(tiled=args.tiled, imgsz=args.imgsz))

    for cfg in configs:
        cfg.update(
            tile=args.tile,
            overlap=args.overlap,
            merge_dist=args.merge_dist,
            include_full=not args.no_full_pass,
            backend=args.backend,
        )
        tag = tag_for(cfg)
        print(f"\n[{tag}]")

        rows, scores, secs = run_one(model, args.weights, df, img_dir, cfg)

        loc_path = C.RESULTS / f"pred_loc_{args.split}_{tag}.csv"
        ocxr.write_localization_csv(loc_path, rows)

        pd.DataFrame(
            {"image_name": df.image_name, "score": scores, "truth": df.has_object.values}
        ).to_csv(C.RESULTS / f"pred_cls_{args.split}_{tag}.csv", index=False)

        n_pred = sum(len(p) for _, p in rows)
        (C.RESULTS / f"predict_{args.split}_{tag}.json").write_text(
            json.dumps(
                {
                    "weights": str(args.weights),
                    "split": args.split,
                    "n_images": len(df),
                    "config": {k: v for k, v in cfg.items()},
                    "conf_floor": C.PRED_CONF,
                    "total_predictions": n_pred,
                    "mean_predictions_per_image": round(n_pred / max(1, len(rows)), 2),
                    "seconds": round(secs, 1),
                },
                indent=2,
            )
        )

        print(f"    {n_pred} predictions ({n_pred / max(1, len(rows)):.1f}/image) in {secs / 60:.1f} min")
        print(f"    -> {loc_path.name}")

        if n_pred == 0:
            print("    !! zero predictions. Check the weights loaded and conf floor is 0.001.")

    print("\nNext: python evaluate.py --split", args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
