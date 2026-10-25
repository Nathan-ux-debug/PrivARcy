"""CLI: merge several separate single-class Roboflow-style exports into
one unified raw dataset, ready for dataset_pipeline.cli's normal split.

Example — exactly your situation, 6 separately-downloaded per-class
exports sitting in sibling folders:

    python -m dataset_pipeline.merge_cli \\
        --class-dir id_card=datasets\\id_card \\
        --class-dir document=datasets\\document \\
        --class-dir screen=datasets\\screen \\
        --class-dir face=datasets\\face \\
        --class-dir credit_card=datasets\\credit_card \\
        --class-dir license_plate=datasets\\license_plate \\
        --output-dir datasets\\merged_raw

Then feed the result into the normal split step:

    python -m dataset_pipeline.cli \\
        --images-dir datasets\\merged_raw\\images \\
        --labels-dir datasets\\merged_raw\\labels \\
        --classes id_card,document,screen,face,credit_card,license_plate \\
        --output-dir dataset_final
"""

import argparse
import logging
import sys

from .exceptions import DatasetValidationError
from .models import ClassMap
from .merge_exports import merge_class_exports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge several separate single-class dataset exports into one unified raw dataset."
    )
    parser.add_argument(
        "--class-dir", action="append", required=True, metavar="CLASS_NAME=PATH",
        help="Repeatable. e.g. --class-dir credit_card=datasets/credit_card",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    classes = parser.add_mutually_exclusive_group(required=True)
    classes.add_argument("--classes", type=str, help="Comma-separated FULL class list, in the order you want class ids assigned.")
    classes.add_argument("--classes-file", type=str, help="Path to a text file, one class name per line.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    class_export_dirs = {}
    for entry in args.class_dir:
        if "=" not in entry:
            logger.error("--class-dir entries must be CLASS_NAME=PATH, got: %s", entry)
            return 1
        name, path = entry.split("=", 1)
        class_export_dirs[name.strip()] = path.strip()

    class_map = ClassMap(names=[c.strip() for c in args.classes.split(",")]) if args.classes else ClassMap.from_file(args.classes_file)

    try:
        stats = merge_class_exports(class_export_dirs, args.output_dir, class_map)
    except DatasetValidationError as exc:
        logger.error("Merge failed: %s", exc)
        return 1

    logger.info("Done: %s", stats.to_dict())
    logger.info(
        "Next: python -m dataset_pipeline.cli --images-dir %s/images --labels-dir %s/labels "
        "--classes %s --output-dir <final_dataset_dir>",
        args.output_dir, args.output_dir, ",".join(class_map.names),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
