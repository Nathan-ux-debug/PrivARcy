# Face modes

| Mode | Rule | New people who appear later |
|---|---|---|
| **Auto** (recommended) | Everyone is blurred except people you keep visible. | Blurred automatically. |
| **Selected only** | Only people you select are blurred. Everyone else is left alone by this rule. | Left alone until you select them. |
| **Every face** | Every detected face is blurred. Choices are ignored. | Blurred automatically. |

The modes are mutually exclusive. Other classes (ID cards, screens, plates, ...) follow their own chips in every mode.

## How people are told apart

1. Within one continuous appearance, the same person is followed by position (tracking).
2. A person who leaves and returns is re-linked by a HOG appearance signature (`FaceEmbedder`) against the `FaceBook`.
3. Two faces in the same frame are never the same person.

The HOG signature is a baseline, not deep face recognition. So:

* Linking is **strict** in Auto mode (a different-looking face is never treated as the kept-visible person).
* Linking is **loose** in Selected mode (a selected person is more likely to stay blurred).
* A face that is too small (< 24 px) or unreadable is "unidentified" and is blurred in every mode.

The practical effect: mistakes lean towards blurring. A person who leaves and returns may appear twice in the picker.

## Where it lives

`faces/identity.py` (FaceBook), `faces/policy.py` (FacePolicy, FaceMode), `faces/scanner.py` (video scan),
`pipeline/frame_engine.py` (applies the verdict), `pipeline/preview.py` (same decision for the editor).

## Upgrading accuracy

`FaceBook` only needs an object with `embed(crop) -> vector`. A deep embedder can replace `FaceEmbedder`; keep the two
thresholds in `faces/identity.py` in mind when you do.
