"""
convert_polygon_to_bbox.py

Converts polygon-format YOLO label lines into bounding-box format by
computing the tightest enclosing rectangle around each polygon's
vertices — no manual redrawing needed.

This is exactly what a "Convert to Box" button does in an annotation
tool, just run locally against your already-downloaded label files:
for each polygon (class_id + a long list of x,y vertex pairs), the
output box is [min(x), min(y)] to [max(x), max(y)] across every vertex,
converted to YOLO's (x_center, y_center, width, height) format.

Already-bbox lines are left completely untouched, so this is safe to
run on a folder with a MIX of bbox and polygon lines (handles the
"mixed" file case from check_label_format.py correctly) — only the
polygon lines get converted, everything else passes through as-is.

Usage:
    # Convert in place (overwrites the original .txt files)
    python convert_polygon_to_bbox.py datasets\\screen

    # Write converted copies to a separate folder instead (safer first run)
    python convert_polygon_to_bbox.py datasets\\screen --output-dir datasets\\screen_converted

    # Preview what would change without writing anything
    python convert_polygon_to_bbox.py datasets\\screen --dry-run
"""

import argparse
import sys
from pathlib import Path


def classify_and_convert_line(line: str) -> tuple[str, str]:
    """Returns (status, output_line).

    status is one of: "bbox" (unchanged), "converted" (was polygon, now
    bbox), "malformed" (left as-is, couldn't safely convert).
    """
    parts = line.split()
    n = len(parts)

    if n == 5:
        return "bbox", line  # already correct, pass through unchanged

    if n > 5 and (n - 1) % 2 == 0:
        try:
            class_id = parts[0]
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            return "malformed", line

        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # Clamp — polygon vertices are already normalized [0,1] in valid
        # YOLO exports, but guard against tiny floating-point overshoot.
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)

        new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        return "converted", new_line

    return "malformed", line


def convert_file(path: Path) -> tuple[str, int, int, int]:
    """Returns (new_content, bbox_count, converted_count, malformed_count)."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]

    out_lines = []
    counts = {"bbox": 0, "converted": 0, "malformed": 0}
    for line in lines:
        status, new_line = classify_and_convert_line(line)
        counts[status] += 1
        out_lines.append(new_line)

    content = "\n".join(out_lines) + ("\n" if out_lines else "")
    return content, counts["bbox"], counts["converted"], counts["malformed"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert polygon YOLO labels to bounding boxes (tightest enclosing rectangle).")
    parser.add_argument("root", type=str, help="Folder to scan recursively for .txt label files.")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Write converted files here (mirroring folder structure) instead of overwriting in place.")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing any files.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Path not found: {root}")
        return 1

    txt_files = [p for p in root.rglob("*.txt") if not p.name.upper().startswith("README")]
    if not txt_files:
        print(f"No label .txt files found under {root}")
        return 1

    output_root = Path(args.output_dir) if args.output_dir else None

    total_bbox = total_converted = total_malformed = 0
    files_changed = 0

    for path in sorted(txt_files):
        content, bbox_n, converted_n, malformed_n = convert_file(path)
        total_bbox += bbox_n
        total_converted += converted_n
        total_malformed += malformed_n

        if converted_n > 0:
            files_changed += 1
            if not args.dry_run:
                if output_root:
                    dest = output_root / path.relative_to(root)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content)
                else:
                    path.write_text(content)

    mode = "DRY RUN — nothing written" if args.dry_run else ("Written to " + str(output_root.resolve()) if output_root else "Overwritten in place")
    print(f"Scanned {len(txt_files)} file(s), {files_changed} file(s) had at least one polygon line.")
    print(f"  bbox lines (unchanged):  {total_bbox}")
    print(f"  polygon lines converted: {total_converted}")
    print(f"  malformed lines (left untouched, couldn't convert): {total_malformed}")
    print(f"Mode: {mode}")

    if total_malformed:
        print(f"\n{total_malformed} malformed line(s) remain — these need manual attention "
              f"(re-run check_label_format.py to find exactly which files).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
