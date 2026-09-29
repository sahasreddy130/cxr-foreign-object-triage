"""
Phase 4 — build the object-category layer object-CXR does not ship with.

    python annotate.py --make-crops     # one PNG per annotated object
    python annotate.py --assign         # per-annotator worksheets + overlap set
    #   ... four people fill in the `category` column ...
    python annotate.py --kappa          # inter-rater agreement on the overlap set
    python annotate.py --merge          # -> categories_final.csv
    python annotate.py --stats          # category distribution

Why this exists
---------------
object-CXR annotates *where* foreign objects are, never *what* they are. A
per-category results table therefore cannot be produced from the dataset alone —
which is exactly why the original draft's coins-95% / buttons-92% / needles-87%
table had nothing behind it.

Labelling the categories yourselves fixes that honestly and is the most
defensible original contribution in the project. Three things make it research
rather than opinion, and this script enforces all three:

  1. Every annotator labels a shared overlap set, so agreement is measurable.
  2. Agreement is reported as Fleiss' kappa alongside the results, not hidden.
  3. The object -> category mapping is explicit and machine-checkable, keyed on
     (image_name, object_index) where object_index is position in the CSV's
     ';'-separated list.

Work from the crops, not the full radiographs. A 2000x2500 film takes seconds to
open and pan; a padded crop is instant, and consistency improves when everyone
is looking at the same framing.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config as C
import ocxr

Image.MAX_IMAGE_PIXELS = None

CROPS = C.ANNOTATIONS / "crops"
WORKSHEETS = C.ANNOTATIONS / "worksheets"
TEMPLATE_COLS = ["image_name", "object_index", "annotation_type", "n_objects_in_image",
                 "crop_file", "category", "notes"]


# --------------------------------------------------------------------------


def object_table(split_csv: Path) -> pd.DataFrame:
    """One row per annotated object across the split."""
    df = ocxr.load_split(split_csv)
    rows = []
    for r in df.itertuples(index=False):
        for obj in r.objects:
            rows.append(
                {
                    "image_name": r.image_name,
                    "object_index": obj.index,
                    "annotation_type": ocxr.TYPE_NAMES[obj.type],
                    "n_objects_in_image": r.n_objects,
                    "area_px": round(obj.area(), 1),
                }
            )
    return pd.DataFrame(rows)


def make_crops(split_csv: Path, img_dir: Path, pad: float) -> pd.DataFrame:
    """Save a padded crop per object. Returns the object table with crop paths."""
    CROPS.mkdir(parents=True, exist_ok=True)
    df = ocxr.load_split(split_csv)
    rows = []

    positives = df[df.n_objects > 0]
    for k, r in enumerate(positives.itertuples(index=False), 1):
        with Image.open(img_dir / r.image_name) as im:
            im = im.convert("L")
            W, H = im.size
            for obj in r.objects:
                x1, y1, x2, y2 = obj.to_bbox()
                # Pad proportionally to the object, with a floor so that tiny
                # objects still get enough surrounding anatomy to be identifiable.
                px = max((x2 - x1) * pad, 60.0)
                py = max((y2 - y1) * pad, 60.0)
                box = (
                    int(max(0, x1 - px)),
                    int(max(0, y1 - py)),
                    int(min(W, x2 + px)),
                    int(min(H, y2 + py)),
                )
                if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                    continue
                stem = Path(r.image_name).stem
                fname = f"{stem}_{obj.index}.png"
                crop = im.crop(box)
                # Cap the long side; some polygons span half the film.
                crop.thumbnail((512, 512))
                crop.save(CROPS / fname)
                rows.append(
                    {
                        "image_name": r.image_name,
                        "object_index": obj.index,
                        "annotation_type": ocxr.TYPE_NAMES[obj.type],
                        "n_objects_in_image": r.n_objects,
                        "crop_file": fname,
                        "category": "",
                        "notes": "",
                    }
                )
        if k % 100 == 0:
            print(f"  {k}/{len(positives)} images", flush=True)

    return pd.DataFrame(rows)


def assign(table: pd.DataFrame, n_annotators: int, overlap_n: int, seed: int) -> dict[str, pd.DataFrame]:
    """Split the objects into per-annotator worksheets plus a shared overlap set.

    The overlap set is drawn at the level of *images*, not objects, so an image
    is never split across annotators — context matters when deciding whether two
    nearby blobs are one necklace or two.
    """
    rng = np.random.default_rng(seed)
    images = table.image_name.unique()
    rng.shuffle(images)

    overlap_imgs = set(images[:overlap_n])
    rest = images[overlap_n:]
    chunks = np.array_split(rest, n_annotators)

    sheets: dict[str, pd.DataFrame] = {}
    for i, chunk in enumerate(chunks, 1):
        mine = table[table.image_name.isin(set(chunk) | overlap_imgs)].copy()
        mine["is_overlap"] = mine.image_name.isin(overlap_imgs)
        mine["annotator"] = f"annotator{i}"
        sheets[f"annotator{i}"] = mine.sort_values(
            ["is_overlap", "image_name", "object_index"], ascending=[False, True, True]
        )
    return sheets


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa for `counts` of shape (n_items, n_categories).

    Each row holds how many raters assigned that item to each category; rows must
    sum to the same number of raters.

    Interpretation (Landis & Koch): <0.20 slight, 0.21-0.40 fair, 0.41-0.60
    moderate, 0.61-0.80 substantial, >0.80 almost perfect. For a taxonomy you
    invented, below about 0.6 means the categories are ambiguous and need
    rewriting — not that your annotators are careless.
    """
    n_items, _ = counts.shape
    n_raters = counts.sum(axis=1)
    if not np.all(n_raters == n_raters[0]):
        raise ValueError("every item must be rated by the same number of raters")
    n = int(n_raters[0])
    if n < 2:
        raise ValueError("need at least 2 raters")

    p_i = (np.sum(counts * (counts - 1), axis=1)) / (n * (n - 1))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / (n_items * n)
    p_e = float(np.sum(p_j**2))
    if np.isclose(p_e, 1.0):
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def cohen_kappa(a: pd.Series, b: pd.Series, categories: list[str]) -> float:
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        if x in idx and y in idx:
            m[idx[x], idx[y]] += 1
    total = m.sum()
    if total == 0:
        return float("nan")
    po = np.trace(m) / total
    pe = float((m.sum(axis=0) / total) @ (m.sum(axis=1) / total))
    return (po - pe) / (1 - pe) if not np.isclose(pe, 1.0) else 1.0


def load_completed() -> pd.DataFrame:
    files = sorted(WORKSHEETS.glob("annotator*_completed.csv"))
    if not files:
        raise SystemExit(
            f"no completed worksheets in {WORKSHEETS}.\n"
            "Each annotator should save theirs as annotator<N>_completed.csv."
        )
    frames = []
    for f in files:
        d = pd.read_csv(f)
        missing = {"image_name", "object_index", "category"} - set(d.columns)
        if missing:
            raise SystemExit(f"{f.name} is missing columns: {sorted(missing)}")
        d["annotator"] = f.stem.split("_")[0]
        blank = d.category.isna() | (d.category.astype(str).str.strip() == "")
        if blank.any():
            print(f"  !! {f.name}: {int(blank.sum())} rows still blank — these are dropped")
            d = d[~blank]
        bad = set(d.category.unique()) - set(C.CATEGORIES)
        if bad:
            raise SystemExit(
                f"{f.name} uses categories outside config.CATEGORIES: {sorted(bad)}\n"
                "Fix the spelling, or add the category to config.py if it is genuinely new."
            )
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def report_kappa(done: pd.DataFrame) -> dict:
    """Agreement on the objects every annotator labelled."""
    key = ["image_name", "object_index"]
    counts_per_obj = done.groupby(key).annotator.nunique()
    shared = counts_per_obj[counts_per_obj == done.annotator.nunique()].index
    if len(shared) == 0:
        raise SystemExit("no objects were labelled by every annotator — was --assign used?")

    sub = done.set_index(key).loc[shared].reset_index()
    n_raters = done.annotator.nunique()

    mat = np.zeros((len(shared), len(C.CATEGORIES)), dtype=int)
    cat_idx = {c: i for i, c in enumerate(C.CATEGORIES)}
    for i, k in enumerate(shared):
        rows = sub[(sub.image_name == k[0]) & (sub.object_index == k[1])]
        for c in rows.category:
            mat[i, cat_idx[c]] += 1

    keep = mat.sum(axis=1) == n_raters
    kappa = fleiss_kappa(mat[keep])

    pairwise = {}
    for a, b in combinations(sorted(done.annotator.unique()), 2):
        da = sub[sub.annotator == a].set_index(key).category
        db = sub[sub.annotator == b].set_index(key).category
        common = da.index.intersection(db.index)
        pairwise[f"{a}|{b}"] = round(cohen_kappa(da.loc[common], db.loc[common], C.CATEGORIES), 4)

    disagreements = []
    for i, k in enumerate(shared):
        rows = sub[(sub.image_name == k[0]) & (sub.object_index == k[1])]
        if rows.category.nunique() > 1:
            disagreements.append(
                {"image_name": k[0], "object_index": int(k[1]),
                 "labels": ", ".join(sorted(rows.category))}
            )

    return {
        "n_overlap_objects": int(keep.sum()),
        "n_raters": int(n_raters),
        "fleiss_kappa": round(float(kappa), 4),
        "pairwise_cohen": pairwise,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements[:100],
    }


def merge(done: pd.DataFrame) -> pd.DataFrame:
    """One category per object: majority vote, ties broken as 'unclear'."""
    key = ["image_name", "object_index"]
    out = []
    for k, grp in done.groupby(key):
        vc = grp.category.value_counts()
        winner = vc.index[0] if (len(vc) == 1 or vc.iloc[0] > vc.iloc[1]) else "unclear"
        out.append(
            {
                "image_name": k[0],
                "object_index": int(k[1]),
                "category": winner,
                "n_labels": int(len(grp)),
                "unanimous": bool(grp.category.nunique() == 1),
            }
        )
    return pd.DataFrame(out).sort_values(key)


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-crops", action="store_true")
    ap.add_argument("--assign", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--annotators", type=int, default=C.N_ANNOTATORS)
    ap.add_argument("--overlap", type=int, default=C.OVERLAP_N)
    args = ap.parse_args()

    if not any([args.make_crops, args.assign, args.kappa, args.merge, args.stats]):
        ap.print_help()
        return 1

    C.seed_everything()
    WORKSHEETS.mkdir(parents=True, exist_ok=True)
    table_path = C.ANNOTATIONS / "object_table.csv"

    if args.make_crops:
        print("cropping objects from the dev (test) split ...")
        t = make_crops(C.RAW_DEV_CSV, C.RAW_DEV_DIR, C.CROP_PAD)
        t.to_csv(table_path, index=False)
        print(f"  {len(t)} objects across {t.image_name.nunique()} images")
        print(f"  crops -> {CROPS}")
        print(f"  table -> {table_path}")

    if args.assign:
        if not table_path.exists():
            raise SystemExit("run --make-crops first")
        t = pd.read_csv(table_path)
        sheets = assign(t, args.annotators, args.overlap, C.SEED)
        for name, sheet in sheets.items():
            p = WORKSHEETS / f"{name}.csv"
            sheet.to_csv(p, index=False)
            n_ov = int(sheet.is_overlap.sum())
            print(f"  {p.name}: {len(sheet)} objects ({n_ov} in the shared overlap set)")
        (WORKSHEETS / "INSTRUCTIONS.txt").write_text(
            "How to label\n"
            "============\n\n"
            f"1. Open your worksheet (annotator<N>.csv) in Sheets or Excel.\n"
            f"2. For each row, open crops/<crop_file> and fill in `category`.\n"
            f"3. Allowed values, spelled exactly:\n"
            + "".join(f"     {c}\n" for c in C.CATEGORIES)
            + "\n"
            "4. Use `unclear` freely. A wrong confident label is worse than an\n"
            "   honest 'unclear', which you can report as a category in its own right.\n"
            "5. Put anything odd in `notes` — those become the qualitative examples.\n"
            "6. Rows with is_overlap=True are labelled by everyone. Do NOT discuss\n"
            "   them with each other before the kappa is computed; that is the whole\n"
            "   point of the overlap set.\n"
            "7. Save as annotator<N>_completed.csv in this folder.\n\n"
            "If you find yourself wanting a category that does not exist, stop and\n"
            "raise it with the group rather than forcing it into `other`. Changing\n"
            "the taxonomy after labelling has started costs everyone a re-pass.\n"
        )
        print(f"  instructions -> {WORKSHEETS / 'INSTRUCTIONS.txt'}")

    if args.kappa:
        done = load_completed()
        rep = report_kappa(done)
        (C.RESULTS / "annotation_agreement.json").write_text(json.dumps(rep, indent=2))
        print(f"\noverlap objects: {rep['n_overlap_objects']}  raters: {rep['n_raters']}")
        print(f"Fleiss' kappa:   {rep['fleiss_kappa']}")
        for pair, k in rep["pairwise_cohen"].items():
            print(f"  Cohen {pair}: {k}")
        print(f"disagreements:   {rep['n_disagreements']}")
        k = rep["fleiss_kappa"]
        if k < 0.4:
            print("\n  !! Below 0.40. The taxonomy is ambiguous. Revise the categories and re-label")
            print("     the overlap set before labelling anything else.")
        elif k < 0.6:
            print("\n  Moderate. Reconcile the disagreements and tighten the category definitions.")
        else:
            print("\n  Substantial or better. Report this figure in the paper.")

    if args.merge:
        done = load_completed()
        final = merge(done)
        p = C.ANNOTATIONS / "categories_final.csv"
        final.to_csv(p, index=False)
        print(f"\n{len(final)} objects -> {p}")
        print(f"  unanimous: {int(final.unanimous.sum())} ({final.unanimous.mean():.1%})")

    if args.stats:
        p = C.ANNOTATIONS / "categories_final.csv"
        if not p.exists():
            raise SystemExit("run --merge first")
        final = pd.read_csv(p)
        counts = final.category.value_counts()
        print("\ncategory distribution")
        for cat, n in counts.items():
            print(f"  {cat:20s} {n:5d}  {n / len(final):6.1%}")
        thin = counts[counts < 20]
        if len(thin):
            print(
                "\n  !! Fewer than 20 objects in: "
                + ", ".join(thin.index)
                + "\n     Per-category sensitivity for these will have very wide confidence"
                "\n     intervals. Either merge them into `other` or report the counts"
                "\n     prominently so nobody over-reads the percentage."
            )
        counts.to_frame("n").to_csv(C.RESULTS / "category_distribution.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
