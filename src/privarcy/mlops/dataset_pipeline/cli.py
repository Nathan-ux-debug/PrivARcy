"""Command-line entry point for the dataset_pipeline package.

Examples:
    # Local raw folder -> 70/15/15 split + data.yaml
    python -m dataset_pipeline.cli \\
        --images-dir raw/images --labels-dir raw/labels \\
        --classes person,car,bicycle \\
        --output-dir dataset

    # Same, but classes come from a file (one name per line)
    python -m dataset_pipeline.cli \\
        --images-dir raw/images --labels-dir raw/labels \\
        --classes-file classes.txt \\
        --output-dir dataset --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

    # Pull from Roboflow instead of a local folder
    python -m dataset_pipeline.cli \\
        --roboflow-api-key $ROBOFLOW_API_KEY --roboflow-workspace my-team \\
        --roboflow-project my-project --roboflow-version 3 \\
        --classes person,car --output-dir dataset
"""

import argparse
import logging
import sys
from pathlib import Path

from .exceptions import DatasetValidationError, DatasetSourceError
from .models import ClassMap
from .sources import LocalDirectorySource, RoboflowSource
from .splitter import DatasetSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, split (train/val/test), and stage a YOLO-format dataset for YOLOv8 training."
    )

    local = parser.add_argument_group("Local directory source")
    local.add_argument("--images-dir", type=str, help="Folder of raw images.")
    local.add_argument("--labels-dir", type=str, help="Folder of matching YOLO .txt label files.")

    roboflow = parser.add_argument_group("Roboflow source (requires `pip install roboflow`)")
    roboflow.add_argument("--roboflow-api-key", type=str, help="Roboflow API key.")
    roboflow.add_argument("--roboflow-workspace", type=str, help="Roboflow workspace slug.")
    roboflow.add_argument("--roboflow-project", type=str, help="Roboflow project slug.")
    roboflow.add_argument("--roboflow-version", type=int, help="Roboflow dataset version number.")

    classes = parser.add_mutually_exclusive_group(required=True)
    classes.add_argument("--classes", type=str, help="Comma-separated class names, in class-id order.")
    classes.add_argument("--classes-file", type=str, help="Path to a text file, one class name per line.")

    parser.add_argument("--output-dir", type=str, required=True, help="Where to write the split dataset + data.yaml.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed, for a reproducible split.")
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move"])
    parser.add_argument("--allow-missing-labels", action="store_true",
                         help="Treat images with no label file as background examples instead of excluding them.")
    parser.add_argument("--no-strict", action="store_true",
                         help="Don't abort on orphan/malformed labels — exclude them and proceed.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.classes:
        class_map = ClassMap(names=[c.strip() for c in args.classes.split(",") if c.strip()])
    else:
        class_map = ClassMap.from_file(args.classes_file)

    using_local = bool(args.images_dir or args.labels_dir)
    using_roboflow = bool(args.roboflow_api_key or args.roboflow_workspace or args.roboflow_project)

    if using_local and using_roboflow:
        logger.error("Pass either --images-dir/--labels-dir OR the --roboflow-* options, not both.")
        return 1

    if using_local:
        if not (args.images_dir and args.labels_dir):
            logger.error("Local source requires both --images-dir and --labels-dir.")
            return 1
        source = LocalDirectorySource(args.images_dir, args.labels_dir)
    elif using_roboflow:
        missing = [name for name, val in [
            ("--roboflow-api-key", args.roboflow_api_key),
            ("--roboflow-workspace", args.roboflow_workspace),
            ("--roboflow-project", args.roboflow_project),
            ("--roboflow-version", args.roboflow_version),
        ] if not val]
        if missing:
            logger.error("Roboflow source is missing required option(s): %s", ", ".join(missing))
            return 1
        source = RoboflowSource(
            api_key=args.roboflow_api_key,
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
            version=args.roboflow_version,
        )
    else:
        logger.error("Provide a dataset source: --images-dir/--labels-dir or the --roboflow-* options.")
        return 1

    splitter = DatasetSplitter(
        source=source,
        output_dir=Path(args.output_dir),
        class_names=class_map,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        mode=args.mode,
        allow_missing_labels=args.allow_missing_labels,
        strict_validation=not args.no_strict,
    )

    try:
        stats = splitter.run()
    except (DatasetValidationError, DatasetSourceError) as exc:
        logger.error("Dataset pipeline failed: %s", exc)
        return 1

    logger.info("Summary: %s", stats.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
