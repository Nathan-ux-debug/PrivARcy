"""Diagnose the 'imported OpenCV module does not provide CascadeClassifier' error.

Run with the SAME interpreter that launches the app:
    python scripts\\diagnostics\\diagnose_opencv.py

It only READS the environment. It installs/uninstalls nothing.
"""
import importlib.metadata as md
import os
import sys

print("interpreter     :", sys.executable)
print("python          :", sys.version.split()[0])
print("working dir     :", os.getcwd())

# 1. Is something named cv2 shadowing the real package?
shadow = []
for entry in sys.path:
    for name in ("cv2", "cv2.py"):
        p = os.path.join(entry or os.getcwd(), name)
        if os.path.exists(p):
            shadow.append(p)
print("cv2 candidates on sys.path:")
for p in shadow:
    print("   ", p)

# 2. Which OpenCV distributions are installed (conflicts = more than one)?
names = ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless")
found = []
for n in names:
    try:
        found.append((n, md.version(n)))
    except md.PackageNotFoundError:
        pass
print("OpenCV distributions:", found or "none")
if len(found) > 1:
    print("  >> WARNING: several OpenCV distributions are installed; they overwrite each other's cv2 files.")
if any(v.startswith("5") for _, v in found):
    print("  >> NOTE: OpenCV 5.x reorganised modules; classical Haar cascades may not be available there.")

# 3. What actually imported?
import cv2  # noqa: E402

print("cv2.__file__    :", getattr(cv2, "__file__", None))
print("cv2.__version__ :", getattr(cv2, "__version__", None))
for attr in ("CascadeClassifier", "HOGDescriptor", "FaceDetectorYN", "dnn_TextDetectionModel_DB", "VideoCapture", "VideoWriter"):
    print(f"  cv2.{attr:<28}", "OK" if hasattr(cv2, attr) else "MISSING")
data = getattr(cv2, "data", None)
print("  cv2.data.haarcascades       ", getattr(data, "haarcascades", "MISSING"))

print()
if hasattr(cv2, "CascadeClassifier"):
    print("RESULT: this interpreter's OpenCV provides CascadeClassifier. The error comes from a DIFFERENT interpreter/environment.")
    print("        Launch the app with the interpreter printed above.")
else:
    print("RESULT: CascadeClassifier is missing here. See the distributions/version above; fix that package set once, deliberately.")
print("NOTE  : PrivARcy now starts either way: YOLO face boxes are used when Haar is unavailable.")
