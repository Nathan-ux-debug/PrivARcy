# Model weights

`best.pt` - the trained YOLO detector (classes: `id_card`, `document`, `screen`, `face`, `credit_card`, `license_plate`).

* Place the file here (`models/best.pt`), or point `PRIVARCY_MODELS` at another folder.
* Weights are large, so they are git-ignored. To version them, use Git LFS:
  `git lfs track "models/*.pt"` and remove the `/models/*.pt` line from `.gitignore`.
* Sensitive text (OCR) additionally needs `pip install -e ".[ocr]"` (Hugging Face `transformers` + `torch`).
