"""Command-line entry point for the active_learning package.

Example:
    python -m active_learning.cli \\
        --corrections-db correction_log.db --frames-dir frames_out \\
        --classes id_card,document,screen,face,credit_card,license_plate \\
        --output-dir retrain_batch

The output (retrain_batch/images, retrain_batch/labels) is ready to feed
straight into dataset_pipeline, either as its own --images-dir/--labels-dir
run, or merged into your main annotated dataset folder before splitting.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from data_store import CorrectionLogDB
from dataset_pipeline import ClassMap

from .exporter import CorrectionExporter
from .exceptions import ActiveLearningError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export reviewer-confirmed corrections as new YOLO training data.")
    parser.add_argument("--corrections-db", type=str, required=True)
    parser.add_argument("--frames-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    classes = parser.add_mutually_exclusive_group(required=True)
    classes.add_argument("--classes", type=str, help="Comma-separated class names, matching the main dataset's class order.")
    classes.add_argument("--classes-file", type=str, help="Path to a text file, one class name per line.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    class_map = ClassMap(names=[c.strip() for c in args.classes.split(",")]) if args.classes else ClassMap.from_file(args.classes_file)

    db = CorrectionLogDB(args.corrections_db)
    try:
        exporter = CorrectionExporter(db, frames_dir=args.frames_dir, class_map=class_map)
        stats = exporter.export(args.output_dir)
    except ActiveLearningError as exc:
        logger.error("Export failed: %s", exc)
        return 1
    finally:
        db.close()

    logger.info("Done: %s", stats.to_dict())
    if stats.hard_negatives:
        report_path = Path(args.output_dir) / "hard_negatives_report.json"
        report_path.write_text(json.dumps(stats.hard_negatives, indent=2))
        logger.info("Wrote %d hard-negative case(s) for manual review to %s", len(stats.hard_negatives), report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
