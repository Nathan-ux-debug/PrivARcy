"""Generates the data.yaml file Ultralytics YOLOv8 training expects.

Written by hand (not via PyYAML) to keep this package's dependencies down
to just the Python standard library — the YAML this needs is simple and
flat, so a hand-rolled writer avoids an extra dependency for something
this small.
"""

from pathlib import Path
from typing import List

from .models import ClassMap


def write_data_yaml(
    output_dir: Path,
    class_map: ClassMap,
    yaml_path: Path | None = None,
    include_test: bool = True,
) -> Path:
    """Write a data.yaml describing the split dataset at output_dir.

    Args:
        output_dir: Root of the split dataset — expected to contain
                     images/train, images/val, (images/test) subfolders.
        class_map: Class names, in id order.
        yaml_path: Where to write the file. Defaults to output_dir/data.yaml.
        include_test: Whether to include a `test:` key (omit if you only
                       have train/val, e.g. a very small dataset).

    Returns:
        Path to the written data.yaml file.
    """
    output_dir = Path(output_dir).resolve()
    yaml_path = Path(yaml_path) if yaml_path else output_dir / "data.yaml"

    names_block = "\n".join(f"  {i}: {_yaml_scalar(name)}" for i, name in enumerate(class_map.names))

    lines: List[str] = [
        f"path: {_yaml_scalar(str(output_dir))}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.append(f"nc: {class_map.num_classes}")
    lines.append("names:")
    lines.append(names_block)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def _yaml_scalar(value: str) -> str:
    """Quote a scalar string for YAML if it contains characters that
    would otherwise change its meaning (colons, quotes, leading/trailing
    whitespace, etc.). Good enough for class names and filesystem paths;
    not a general-purpose YAML emitter."""
    needs_quoting = (
        value == ""
        or value != value.strip()
        or any(ch in value for ch in [":", "#", "'", '"', "{", "}", "[", "]", ",", "&", "*"])
    )
    if needs_quoting:
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value
