# dataset_pipeline

Validates a raw folder of annotated images + YOLO label files, splits it
into train/val/test, lays the result out in the directory structure
**Ultralytics YOLOv8** expects for fine-tuning, and writes the
`data.yaml` training config. Also supports pulling the raw dataset from
**Roboflow** instead of a local folder, so a team can point at the same
shared source.

This is Ticket 3 of the pipeline:

```
frame_extraction → (external annotation step) → raw images + YOLO labels
    → dataset_pipeline (this package) → images/{train,val,test},
       labels/{train,val,test}, data.yaml → Ultralytics YOLOv8 fine-tuning
```

---

## 1. Layout

```
dataset_pipeline/
├── __init__.py        # public API exports
├── exceptions.py        # DatasetValidationError, DatasetSourceError
├── models.py              # ClassMap, SplitCounts, DatasetStats
├── validator.py             # checks raw images/labels are consistent & well-formed
├── sources.py                  # LocalDirectorySource, RoboflowSource
├── splitter.py                    # DatasetSplitter — the core class
├── yaml_writer.py                    # writes data.yaml
├── cli.py                               # command-line entry point
└── tests/
    └── test_dataset_pipeline.py
```

### Why split like this

| File | Responsibility | Why it's separate |
|---|---|---|
| `validator.py` | Confirms every image has a matching label, every label parses correctly, class ids are in range, coordinates are normalized | This is the thing most likely to catch a bad annotation export *before* it wastes a training run. Kept independent so it can be run/tested on its own |
| `sources.py` | Resolves "where does the raw data come from" — a local folder, or a Roboflow project download | Ticket asked for both a local-directory loader and Roboflow integration; this file is the seam between them, so `DatasetSplitter` never needs to know which one it got |
| `splitter.py` | Shuffles (with a seed), cuts train/val/test, copies or moves files into the YOLO layout | The only file that touches the filesystem for the actual dataset files |
| `yaml_writer.py` | Emits `data.yaml` | Small and self-contained — no need for a PyYAML dependency for something this simple |
| `models.py` | `ClassMap` (id↔name), `DatasetStats` (counts returned after a run) | The "contract" other tooling (a training script, a dashboard) can depend on without importing validation/splitting logic |

---

## 2. How it works

### 2.1 Expected raw input

Before running this, you need a **flat** folder of images and a flat
folder of matching YOLO-format `.txt` label files — same base filename,
different extension:

```
raw_images/
  photo001.jpg
  photo002.jpg
  ...
raw_labels/
  photo001.txt
  photo002.txt
  ...
```

Each label file has one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
all four numbers normalized to `[0, 1]` — the standard YOLO format
produced by most annotation tools (LabelImg, CVAT, Roboflow, etc.).

An image with **no** label file is either an error (default) or a valid
background/negative example if you pass `allow_missing_labels=True`.

### 2.2 Validation (`validator.py`)

`validate_raw_dataset()` checks, in order:
1. Both folders exist and contain at least one image.
2. Every image either has a matching label file, or is flagged as
   `stems_without_labels` (excluded unless `allow_missing_labels=True`).
3. Every label file has a matching image — otherwise it's an
   `orphan_label` (a label with no image is almost always a bug in an
   export/sync step).
4. Every label file parses: exactly 5 space-separated values per line,
   class id within the known class range, all 4 coordinates in `[0, 1]`.
   Anything that fails is recorded in `malformed_labels` with the
   specific line and reason.

By default (`strict=True`), any orphan or malformed label **stops the
run** — better to catch a bad export now than train on corrupted data.
Pass `strict=False` to log the problems, exclude the bad entries, and
proceed anyway (useful for a first look at a messy dataset).

### 2.3 Splitting (`splitter.py`)

1. Takes the list of valid image/label stems from validation.
2. Shuffles them with `random.Random(seed)` — same seed always produces
   the same split, so re-running the pipeline on the same raw data
   later (e.g. after adding more images) still gives a fair, comparable
   evaluation split. Default `seed=42`.
3. Cuts the shuffled list at `round(n * train_ratio)` and
   `round(n * train_ratio) + round(n * val_ratio)` — whatever's left
   over from rounding lands in `test`, so counts always sum exactly to
   the total.
4. For each split, creates `images/<split>/` and `labels/<split>/` under
   your output directory and copies (or moves, with `mode="move"`) the
   matching files in. A background image (no label) gets an **empty**
   `.txt` file written for it — YOLO's convention for "no objects here."
5. Calls `write_data_yaml()` to emit `data.yaml`.

### 2.4 `data.yaml` output

```yaml
path: /absolute/path/to/output_dir
train: images/train
val: images/val
test: images/test
nc: 3
names:
  0: person
  1: car
  2: dog
```

This is exactly the file `ultralytics.YOLO(...).train(data="data.yaml")`
expects — `nc` and `names` come straight from the `ClassMap` you pass in,
so there's no way for the class list used to split the data to drift
from the class list written into the config.

### 2.5 Sources — local folder or Roboflow

```python
from dataset_pipeline import LocalDirectorySource, RoboflowSource

# Option A: dataset already on disk
source = LocalDirectorySource("raw_images", "raw_labels")

# Option B: pull from a shared Roboflow project
source = RoboflowSource(
    api_key="...",
    workspace="my-team",
    project="my-project",
    version=3,
)
```

Both resolve to the same thing — a `(images_dir, labels_dir)` pair — so
`DatasetSplitter` doesn't care which one it's given. `RoboflowSource`
downloads via the Roboflow API (`pip install roboflow` required) and,
since Roboflow exports already come pre-split into train/valid/test,
**flattens them back into one raw folder** first — that way this
pipeline's own seeded 70/15/15 split (not Roboflow's) is what actually
determines the train/val/test boundary, so everyone on the team gets the
same, reproducible split regardless of where the raw data came from.

> **Note on this development environment:** `roboflow` isn't installed
> here (network access in this sandbox doesn't include roboflow.com,
> and there's no API key to test against). `RoboflowSource` is written
> against Roboflow's documented Python SDK, and the "missing dependency"
> and "flatten a pre-split export" code paths are unit-tested directly.
> The actual live download is untested — recommend a smoke test against
> a real Roboflow project before relying on it for team sync.

---

## 3. Install

```bash
pip install opencv-python  # only needed if you're feeding this from frame_extraction
pip install roboflow        # only needed for RoboflowSource
```

`dataset_pipeline` itself has **no required third-party dependencies** —
just the Python standard library.

---

## 4. Usage

### 4.1 Library

```python
from dataset_pipeline import DatasetSplitter, LocalDirectorySource, ClassMap

source = LocalDirectorySource("raw_images", "raw_labels")
class_map = ClassMap(names=["person", "car", "dog"])

splitter = DatasetSplitter(
    source=source,
    output_dir="dataset",
    class_names=class_map,
    split_ratios=(0.70, 0.15, 0.15),
    seed=42,
)
stats = splitter.run()
print(stats.to_dict())
```

### 4.2 CLI

```bash
# Local raw folder
python -m dataset_pipeline.cli \
    --images-dir raw_images --labels-dir raw_labels \
    --classes person,car,dog \
    --output-dir dataset

# Classes from a file instead of a comma list
python -m dataset_pipeline.cli \
    --images-dir raw_images --labels-dir raw_labels \
    --classes-file classes.txt \
    --output-dir dataset --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

# From Roboflow
python -m dataset_pipeline.cli \
    --roboflow-api-key $ROBOFLOW_API_KEY --roboflow-workspace my-team \
    --roboflow-project my-project --roboflow-version 3 \
    --classes person,car --output-dir dataset
```

### 4.3 Then train with Ultralytics

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="dataset/data.yaml", epochs=50)
```

---

## 5. Error handling

- Missing images/labels folder, or zero images found → `DatasetValidationError`
  immediately, before any files are touched.
- Orphan labels or malformed label syntax → `DatasetValidationError` in
  strict mode (default); logged and excluded in non-strict mode.
- Invalid `split_ratios` (negative, or don't sum to ~1.0) or an invalid
  `mode` → `ValueError` at `DatasetSplitter` construction, before
  validation even runs.
- Roboflow source: missing `roboflow` package, failed download, or an
  export that doesn't match the expected train/valid/test layout →
  `DatasetSourceError`.

## 6. Tests

```bash
python -m unittest dataset_pipeline.tests.test_dataset_pipeline -v
```

24 tests covering: validation (valid dataset, missing dirs, orphan
labels in strict/non-strict mode, out-of-range class ids, out-of-bounds
coordinates, missing-label handling with and without
`allow_missing_labels`), splitting (correct 70/15/15 counts, correct
directory layout, copy vs. move mode, determinism with a fixed seed,
different seeds producing different splits, invalid ratios/mode
rejected, background images getting empty label files), `data.yaml`
content, `LocalDirectorySource` and `RoboflowSource` resolution
(including the missing-dependency error path), and `ClassMap` lookups.
