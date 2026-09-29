"""
object-CXR annotation model.

The dataset encodes every annotated foreign object as a single string:

    <type> x1 y1 x2 y2 [... xn yn]

with objects separated by ';' and an empty field meaning "no foreign object".
Three types are used:

    0  rectangle   x1 y1 x2 y2  (upper-left, lower-right)
    1  ellipse     x1 y1 x2 y2  (bounding box of the ellipse)
    2  polygon     x1 y1 ... xn yn

Coordinates are absolute pixels, origin upper-left, x rightwards, y downwards.

Why this module exists
----------------------
Two different geometries are needed, and conflating them is the single easiest
way to get this project wrong:

  * TRAINING consumes axis-aligned boxes, because that is what YOLO accepts.
    `to_bbox` collapses each object to its enclosing box. For rectangles and
    ellipses this is exact; for polygons it is lossy and inflates area.

  * EVALUATION must use the true shapes. `contains` implements the official
    containment test — a bounds check for rectangles, the ellipse equation for
    ellipses, and a point-in-polygon test for polygons. This mirrors froc.py
    exactly and is why FROC, not mAP, is the correct metric here: the dataset
    authors state that mixed box/ellipse/polygon annotations make mAP unsuitable.

Never evaluate against `to_bbox` output. It will flatter or punish the model for
reasons that have nothing to do with the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

RECTANGLE, ELLIPSE, POLYGON = "0", "1", "2"
TYPE_NAMES = {RECTANGLE: "rectangle", ELLIPSE: "ellipse", POLYGON: "polygon"}


@dataclass(frozen=True)
class Annotation:
    """One annotated foreign object."""

    image_name: str
    index: int              # position within the image's ';'-separated list
    type: str               # '0' | '1' | '2'
    coords: np.ndarray      # flat float array, length 4 or 2n

    # -- geometry ---------------------------------------------------------

    def to_bbox(self) -> tuple[float, float, float, float]:
        """Enclosing axis-aligned box as (x1, y1, x2, y2).

        Exact for rectangles and ellipses. Lossy for polygons — a chain lying
        diagonally across the lung field yields a box far larger than the object.
        """
        xs, ys = self.coords[0::2], self.coords[1::2]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def contains(self, x: float, y: float) -> bool:
        """True if point (x, y) lies inside this object's true shape.

        Faithful reimplementation of `inside_object` in the challenge's froc.py.
        evaluate.py cross-checks its FROC against the official script; if you
        edit this, that check will fail, which is the point.
        """
        if self.type == RECTANGLE:
            x1, y1, x2, y2 = self.coords
            return bool(x1 <= x <= x2 and y1 <= y <= y2)

        if self.type == ELLIPSE:
            x1, y1, x2, y2 = self.coords
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax, ay = (x2 - x1) / 2.0, (y2 - y1) / 2.0
            if ax == 0 or ay == 0:
                return False
            return bool(((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2 <= 1.0)

        if self.type == POLYGON:
            pts = self.coords.reshape(-1, 2)
            return _point_in_polygon(x, y, pts)

        raise ValueError(f"unknown annotation type {self.type!r}")

    def area(self) -> float:
        """Area of the enclosing box, in square pixels. Used for size analysis."""
        x1, y1, x2, y2 = self.to_bbox()
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


try:
    from skimage.measure import points_in_poly as _sk_points_in_poly
except ImportError:  # pragma: no cover
    _sk_points_in_poly = None


def _point_in_polygon(x: float, y: float, pts: np.ndarray) -> bool:
    """Point-in-polygon test, delegating to skimage where available.

    The challenge's froc.py uses `skimage.measure.points_in_poly`, so we do too.
    This is not incidental. A hand-rolled ray-casting test agrees with skimage on
    simple polygons but can disagree on degenerate ones — self-intersecting
    outlines, repeated vertices, points lying exactly on an edge — and even a
    handful of flipped decisions shifts FROC by a few thousandths. Since the
    entire reason for reporting FROC is comparability with the published
    leaderboard, matching the reference implementation exactly matters more than
    avoiding a dependency.

    The pure-Python fallback exists only so this module still imports without
    skimage; evaluate.py's --verify-froc cross-check will flag any divergence.
    """
    if len(pts) < 3:
        return False

    if _sk_points_in_poly is not None:
        return bool(_sk_points_in_poly(np.array([[x, y]], dtype=float), pts)[0])

    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            denom = yj - yi
            if denom != 0 and x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_annotation_field(image_name: str, field: object) -> list[Annotation]:
    """Parse one CSV `annotation` cell into a list of Annotation objects.

    Empty, NaN and whitespace-only fields yield [] — these are the true
    negatives, and there are exactly as many of them as positives in every
    official split.
    """
    if field is None or (isinstance(field, float) and np.isnan(field)):
        return []
    text = str(field).strip()
    if text == "" or text.lower() == "nan":
        return []

    out: list[Annotation] = []
    for i, chunk in enumerate(text.split(";")):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        atype = parts[0]
        coords = np.asarray([float(v) for v in parts[1:]], dtype=float)

        if atype in (RECTANGLE, ELLIPSE) and coords.size != 4:
            raise ValueError(
                f"{image_name} object {i}: type {atype} needs 4 coords, got {coords.size}"
            )
        if atype == POLYGON and (coords.size < 6 or coords.size % 2):
            raise ValueError(
                f"{image_name} object {i}: polygon needs an even count >= 6, got {coords.size}"
            )

        out.append(Annotation(image_name, i, atype, coords))
    return out


def load_split(csv_path) -> pd.DataFrame:
    """Load a split CSV, adding parsed objects and a binary label column.

    Returns columns: image_name, annotation, objects, has_object, n_objects.
    """
    df = pd.read_csv(csv_path)
    if list(df.columns[:2]) != ["image_name", "annotation"]:
        raise ValueError(
            f"{csv_path}: expected columns ['image_name', 'annotation'], got {list(df.columns)}"
        )
    df["objects"] = [
        parse_annotation_field(n, a) for n, a in zip(df.image_name, df.annotation)
    ]
    df["n_objects"] = df.objects.map(len)
    df["has_object"] = (df.n_objects > 0).astype(int)
    return df


def iter_objects(df: pd.DataFrame) -> Iterator[Annotation]:
    """Flatten a loaded split into a stream of Annotation objects."""
    for objs in df.objects:
        yield from objs


def type_counts(df: pd.DataFrame) -> dict[str, int]:
    """How many rectangles / ellipses / polygons the split contains."""
    counts: dict[str, int] = {}
    for obj in iter_objects(df):
        counts[TYPE_NAMES[obj.type]] = counts.get(TYPE_NAMES[obj.type], 0) + 1
    return counts


# --------------------------------------------------------------------------
# YOLO label conversion
# --------------------------------------------------------------------------


def to_yolo_lines(
    objects: Sequence[Annotation], img_w: int, img_h: int, class_id: int = 0
) -> list[str]:
    """Convert objects to normalised YOLO label lines for TRAINING ONLY.

    Boxes are clamped to the image and degenerate ones dropped — a handful of
    annotations extend a pixel or two beyond the border, and Ultralytics rejects
    a whole label file if any coordinate falls outside [0, 1].
    """
    lines: list[str] = []
    for obj in objects:
        x1, y1, x2, y2 = obj.to_bbox()
        x1, x2 = max(0.0, x1), min(float(img_w), x2)
        y1, y2 = max(0.0, y1), min(float(img_h), y2)
        w, h = x2 - x1, y2 - y1
        if w <= 1.0 or h <= 1.0:
            continue  # degenerate after clamping
        cx, cy = (x1 + x2) / 2.0 / img_w, (y1 + y2) / 2.0 / img_h
        nw, nh = w / img_w, h / img_h
        if not (0.0 < cx < 1.0 and 0.0 < cy < 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


# --------------------------------------------------------------------------
# Prediction I/O — the exact formats froc.py expects
# --------------------------------------------------------------------------


def write_localization_csv(path, rows: Iterable[tuple[str, list[tuple[float, float, float]]]]) -> int:
    """Write the FROC localisation CSV.

        image_name,prediction
        08001.jpg,0.8123 718.0 1248.0;0.4410 320.5 900.2
        08002.jpg,

    Each prediction is `probability x y` — a POINT, not a box. Images with no
    detections keep the trailing comma; froc.py relies on it.

    Filenames must be bare (`08001.jpg`) and match the ground-truth CSV exactly.
    A full path here is the most common cause of a FROC of 0.0.
    """
    n = 0
    with open(path, "w") as f:
        f.write("image_name,prediction\n")
        for name, preds in rows:
            if "," in name:
                raise ValueError(f"image name contains a comma: {name!r}")
            body = ";".join(f"{p:.6f} {x:.1f} {y:.1f}" for p, x, y in preds)
            f.write(f"{name},{body}\n")
            n += 1
    return n


def read_localization_csv(path) -> dict[str, list[tuple[float, float, float]]]:
    """Read back a localisation CSV into {image_name: [(prob, x, y), ...]}."""
    out: dict[str, list[tuple[float, float, float]]] = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, _, body = line.partition(",")
            preds: list[tuple[float, float, float]] = []
            if body.strip():
                for chunk in body.split(";"):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    p, x, y = (float(v) for v in chunk.split())
                    preds.append((p, x, y))
            out[name] = preds
    return out
