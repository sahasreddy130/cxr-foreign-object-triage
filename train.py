"""
Fine-tune YOLO26 on object-CXR.

    python train.py                      # defaults from config.py
    python train.py --model yolo26m.pt --epochs 150
    python train.py --imgsz 960 --batch 8 --name hires

Notes
-----
YOLO26 (Ultralytics, January 2026) performs end-to-end inference without NMS and
drops Distribution Focal Loss. Two consequences worth knowing:

  * There is no NMS IoU threshold to tune at inference. One fewer confound in the
    resolution experiment, and a cleaner story for FROC — the metric counts every
    predicted point outside an annotation as a false positive, so a model that
    emits one detection per object rather than a suppressed cluster is
    structurally better matched to it.

  * Reported CPU ONNX inference is up to ~43% faster than YOLO11n for the nano
    scale, which matters if you end up arguing deployability in the Discussion.

Weights are copied to Drive when training finishes, and every `--save-period`
epochs while it runs. Colab will recycle the VM without warning; anything left
only in /content is gone.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import config as C


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=C.MODEL, help=f"default {C.MODEL}")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--imgsz", type=int, default=C.TRAIN_IMGSZ)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--name", default=None, help="run name; default derived from model+imgsz")
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu'; default auto")
    ap.add_argument("--save-period", type=int, default=10, help="checkpoint to Drive every N epochs")
    ap.add_argument("--resume", action="store_true", help="resume the last interrupted run")
    return ap.parse_args()


def check_gpu() -> str | None:
    try:
        import torch
    except ImportError:
        raise SystemExit("torch not installed — run: pip install -r requirements.txt")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name} ({mem:.1f} GB)")
        return "0"

    print(
        "\n!! No GPU detected. Training on CPU will take days, not hours.\n"
        "   In Colab: Runtime -> Change runtime type -> T4 GPU -> Save.\n"
    )
    if input("Continue on CPU anyway? [y/N] ").strip().lower() != "y":
        raise SystemExit(1)
    return "cpu"


def main() -> int:
    args = parse_args()
    C.seed_everything()

    if not C.DATA_YAML.exists():
        raise SystemExit(f"{C.DATA_YAML} not found — run prepare_data.py first.")

    device = args.device or check_gpu()
    run_name = args.name or f"{Path(args.model).stem}_{args.imgsz}"

    print(C.describe())
    print(f"\nrun name: {run_name}\ndata:     {C.DATA_YAML}\n")

    from ultralytics import YOLO

    model = YOLO(args.model)

    t0 = time.time()
    model.train(
        data=str(C.DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        seed=C.SEED,
        deterministic=True,
        patience=args.patience,
        project=str(C.RUNS),
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        save_period=args.save_period,
        plots=True,
        val=True,
        cache=True,
        workers=2,
        # Radiographs are not natural images. Two deliberate departures from the
        # Ultralytics defaults, both of which belong in the Methods section:
        #
        #   fliplr=0.5  is kept. Left-right flips produce anatomically implausible
        #               films (dextrocardia), but the objects we care about are
        #               orientation-agnostic and the regulariser is worth more
        #               than the anatomical fidelity here. Revisit if you later
        #               care about laterality.
        #   degrees=5.0 small rotations only. Patients are upright and rotation
        #               beyond a few degrees is unrealistic.
        #   hsv_h=0.0   chest radiographs carry no meaningful hue.
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.3,
        mosaic=1.0,
        close_mosaic=10,
    )
    mins = (time.time() - t0) / 60

    run_dir = C.RUNS / run_name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training finished but {best} is missing")

    dest = C.WEIGHTS / f"{run_name}_best.pt"
    shutil.copy(best, dest)
    for artefact in ("results.csv", "results.png", "args.yaml", "confusion_matrix.png"):
        src = run_dir / artefact
        if src.exists():
            shutil.copy(src, C.RESULTS / f"{run_name}_{artefact}")

    (C.RESULTS / f"{run_name}_train_meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": device,
                "seed": C.SEED,
                "minutes": round(mins, 1),
                "weights": str(dest),
            },
            indent=2,
        )
    )

    print(f"\ndone in {mins:.0f} min")
    print(f"weights  -> {dest}")
    print(f"curves   -> {C.RESULTS / (run_name + '_results.png')}")
    print("\nNext: python predict.py --weights", dest, "--split test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
