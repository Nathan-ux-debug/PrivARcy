"""
check_label_format.py

Scans an entire folder tree of YOLO-format .txt label files, classifies
each as bounding-box format (5 values/line), polygon format (>5 values),
mixed, malformed, or empty — prints a per-class summary table — and can
optionally DELETE the bad ones safely (image + label together, never
just the label alone).

Why image+label together, never just the label:
    Deleting only a bad label file leaves its image in the dataset with
    NO label at all. dataset_pipeline then treats that image as a valid
    "no objects here" background example — which is actively wrong if
    the image really does contain a document/card/screen/etc., just
    annotated badly. That teaches the model the opposite of the truth.
    Deleting the image+label PAIR instead just removes that example
    from the dataset entirely, which is safe.

Usage:
    # Just see the numbers (safe, changes nothing)
    python check_label_format.py datasets

    # Also list every good bbox file (usually too many to be useful)
    python check_label_format.py datasets --show-good

    # Actually delete every non-bbox image+label pair (irreversible!)
    python check_label_format.py datasets --delete
"""

import argparse
import sys
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def classify_line(line: str) -> str:
    parts = line.split()
    n = len(parts)
    if n == 0:
        return "empty_line"
    if n == 5:
        return "bbox"
    if n > 5 and (n - 1) % 2 == 0:
        return "polygon"
    return "malformed"


def classify_file(path: Path) -> str:
    """Returns one of: empty, bbox, polygon, mixed, malformed, unreadable"""
    try:
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return "unreadable"

    lines = [ln.strip() for ln in raw_lines if ln.strip()]
    if not lines:
        return "empty"

    verdicts = {classify_line(ln) for ln in lines}
    if "malformed" in verdicts:
        return "malformed"
    if verdicts == {"bbox"}:
        return "bbox"
    if verdicts == {"polygon"}:
        return "polygon"
    return "mixed"


def find_matching_image(label_path: Path) -> Path | None:
    """Given .../<split>/labels/<name>.txt, find .../<split>/images/<name>.<ext>
    (the standard Roboflow export layout — labels/ and images/ as siblings)."""
    dir_parts = list(label_path.parent.parts)
    for i in range(len(dir_parts) - 1, -1, -1):
        if dir_parts[i].lower() == "labels":
            dir_parts[i] = "images"
            break
    else:
        return None  # no "labels" folder in the path — layout doesn't match what we expect

    images_dir = Path(*dir_parts)
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def class_of(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return "(root)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit (and optionally clean up) YOLO label format issues.")
    parser.add_argument("root", type=str, help="Root folder to scan recursively (e.g. datasets)")
    parser.add_argument("--show-good", action="store_true", help="Also list every good bbox file in the report.")
    parser.add_argument("--delete", action="store_true",
                         help="Actually delete image+label pairs for polygon/mixed/malformed files. "
                              "Without this flag, nothing is ever deleted — dry run only.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Path not found: {root}")
        return 1

    txt_files = [p for p in root.rglob("*.txt") if not p.name.upper().startswith("README")]
    if not txt_files:
        print(f"No label .txt files found under {root}")
        return 1

    print(f"Found {len(txt_files)} label file(s) under {root}. Scanning...\n")

    buckets: dict[str, list[Path]] = {"bbox": [], "polygon": [], "mixed": [], "malformed": [], "empty": [], "unreadable": []}
    for i, path in enumerate(sorted(txt_files), start=1):
        if i % 500 == 0 or i == len(txt_files):
            print(f"  [{i}/{len(txt_files)}] scanned...", flush=True)
        buckets[classify_file(path)].append(path)

    # --- Per-class summary table ------------------------------------------------
    class_names = sorted({class_of(p, root) for p in txt_files})
    categories = ["bbox", "polygon", "mixed", "malformed", "empty", "unreadable"]

    print("\n=== SUMMARY BY CLASS ===")
    header = f"{'class':<18}" + "".join(f"{c:>12}" for c in categories) + f"{'TOTAL':>12}"
    print(header)
    print("-" * len(header))
    grand_totals = {cat: 0 for cat in categories}
    for cls in class_names:
        row_counts = {cat: sum(1 for p in buckets[cat] if class_of(p, root) == cls) for cat in categories}
        for cat in categories:
            grand_totals[cat] += row_counts[cat]
        row = f"{cls:<18}" + "".join(f"{row_counts[cat]:>12}" for cat in categories) + f"{sum(row_counts.values()):>12}"
        print(row)
    print("-" * len(header))
    total_row = f"{'TOTAL':<18}" + "".join(f"{grand_totals[cat]:>12}" for cat in categories) + f"{sum(grand_totals.values()):>12}"
    print(total_row)

    bad_files = buckets["polygon"] + buckets["mixed"] + buckets["malformed"]
    total_bad = len(bad_files)
    print(f"\nSUMMARY: {len(buckets['bbox'])} OK  |  {total_bad} need re-annotation/removal  |  "
          f"{len(buckets['empty'])} empty/background  |  {len(buckets['unreadable'])} unreadable\n")

    # --- Write full file-by-file report -----------------------------------------
    report_path = Path("label_audit_report.txt")
    with report_path.open("w", encoding="utf-8") as report:
        for category in ("polygon", "mixed", "malformed", "unreadable"):
            files = buckets[category]
            if files:
                report.write(f"=== {category.upper()} ({len(files)} file(s)) ===\n")
                for f in files:
                    report.write(f"{f}\n")
                report.write("\n")
        if buckets["empty"]:
            report.write(f"=== EMPTY ({len(buckets['empty'])} file(s)) \u2014 usually fine, not deleted ===\n")
            for f in buckets["empty"]:
                report.write(f"{f}\n")
            report.write("\n")
        if args.show_good:
            report.write(f"=== BBOX / OK ({len(buckets['bbox'])} file(s)) ===\n")
            for f in buckets["bbox"]:
                report.write(f"{f}\n")
    print(f"Full file-by-file list written to: {report_path.resolve()}\n")

    if not bad_files:
        print("Nothing to delete \u2014 no polygon/mixed/malformed files found.")
        return 0

    # --- Dry run: show what WOULD be deleted, per class -------------------------
    pairs_found, pairs_missing_image = [], []
    for label_path in bad_files:
        image_path = find_matching_image(label_path)
        if image_path:
            pairs_found.append((label_path, image_path))
        else:
            pairs_missing_image.append(label_path)

    print(f"Of {total_bad} bad label file(s): {len(pairs_found)} have a matching image "
          f"(safe to delete as a pair), {len(pairs_missing_image)} have NO matching image found "
          f"(label would need manual handling).\n")

    if not args.delete:
        print("This was a DRY RUN \u2014 nothing was deleted. Re-run with --delete to actually remove "
              "these image+label pairs.")
        return 1

    # --- Actually delete, image + label together --------------------------------
    deleted_count = 0
    for label_path, image_path in pairs_found:
        try:
            image_path.unlink()
            label_path.unlink()
            deleted_count += 1
        except OSError as exc:
            print(f"  Could not delete pair for {label_path}: {exc}")

    print(f"\nDeleted {deleted_count} image+label pair(s).")
    if pairs_missing_image:
        print(f"{len(pairs_missing_image)} bad label file(s) had no matching image and were left untouched "
              f"(listed in {report_path.name} under their category \u2014 review these manually).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
