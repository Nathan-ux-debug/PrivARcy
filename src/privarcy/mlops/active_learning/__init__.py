"""
active_learning

Closes the loop your decision_engine's FLAGGED tier already opened:
when a human reviewer confirms a flagged detection was genuinely
sensitive (via data_store.CorrectionLogDB), this package turns that
correction into a properly-formatted labeled training example, ready
to feed back into dataset_pipeline for a re-training round.

This is uncertainty-sampling active learning: the FLAGGED tier already
IS the "ask a human about the cases the model is unsure about" query
strategy — this package is the missing "now use their answer to
retrain" half of the loop.

    decision_engine (FLAGGED) -> human review -> data_store (D4, corrected)
        -> active_learning (this package) -> dataset_pipeline (raw input)
        -> re-trained model -> back into yolo_detection

Public API:
    CorrectionExporter       - pulls corrections, writes YOLO-format labels
    ExportStats                 - summary of what was exported
    ActiveLearningError            - raised for export failures
"""

from .exceptions import ActiveLearningError
from .models import ExportStats
from .exporter import CorrectionExporter

__all__ = ["CorrectionExporter", "ExportStats", "ActiveLearningError"]

__version__ = "1.0.0"
