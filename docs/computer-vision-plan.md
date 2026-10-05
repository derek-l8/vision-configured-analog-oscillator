# Computer-Vision Implementation Plan

## Objective

Given a saved photograph of this fixed breadboard, estimate its visible persistent controls and convert those estimates into parameters for the matching LTspice circuit. The task is intentionally limited to one known physical build. It is not general breadboard recognition.

The ten-image visibility check, synthetic execution tests, and first 40-photo real-image training run are complete. A corrected model passed an eight-photo overfit diagnostic covering all four switch combinations and two potentiometer positions. Evaluation has used training photographs only; no held-out capture session has been tested. The LTspice controller and physical calibration are not implemented.

## Current visual conventions

- Primary source: original iPhone JPEG photographs.
- Canonical input: portrait, with the power module at the top and buzzer at the bottom.
- Model size: `576 x 768`, using aspect-preserving resize and padding when required.
- Expected board rotation: approximately plus or minus 25 degrees in the image plane.
- Expected camera perspective: near-overhead, with mild tilt up to approximately 15 degrees.
- `S_LEFT`: maintained switch on the left in the canonical portrait view.
- `S_RIGHT`: maintained switch on the right in the canonical portrait view.
- Switch actuator toward the power module: ON.
- Switch actuator toward the buzzer: OFF.
- Potentiometer angle: 0 degrees right, 90 degrees up, increasing counterclockwise.
- Potentiometer pointer tip: the pointed end of the orange tape arrow.
- Both switches: blue slider toward the power-module/top end is ON; toward the buzzer/bottom end is OFF.

The electrical mapping from `S_LEFT` and `S_RIGHT` to LTspice parameters `Cadd1` and `Cadd2` will be established separately with a powered-off continuity test. It will not be inferred from image appearance.

## Data design

The physical layout, orange potentiometer pointer, and blue switch markers remain fixed after data collection begins. The primary dataset is photographed unpowered with the momentary button released.

Collection proceeds in stages:

| Stage | Images | Purpose |
|---|---:|---|
| Visibility | 10 | Inspect control detail after exact preprocessing; never used for training metrics |
| Smoke | 40-60 | Prove image loading, annotations, transforms, losses, and the training loop with real labeled photographs |
| Pilot | 120 | Three independent 40-image capture sessions |
| Expanded | 240 | Six 40-image sessions, only if the pilot justifies expansion |

Each full session covers ten potentiometer targets from 45 through 315 degrees in 30-degree increments and all four switch combinations. The exact angle is derived from annotated center and pointer-tip coordinates rather than assumed from the requested capture position.

Capture sessions vary lighting, background, distance, and perspective. Board rotation also varies within sessions. Dataset splits use complete sessions so near-duplicate photographs from one setup cannot appear in both training and evaluation.

The three-session pilot uses `s01` for training, `s02` for validation, and `s03` for the final test. The six-session plan keeps `s02` and `s03` held out and adds `s04`–`s06` to training. Choose the split before import; generating a new capture plan does not change an existing imported manifest. The earlier 40-photo runs used only `s01`.

The [published s01 package](../data/pilot-s01/README.md) contains all 40 processed inputs and their manual labels. Its image paths resolve relative to the manifest, so it can be moved or loaded from another working directory. Use `data/pilot-s01/manifest.json` and `data/pilot-s01/annotations.json` in the training commands below without importing or annotating again. All records are training data; validation/test commands require additional sessions.

## Labels and geometry

The capture manifest records image identity, session, intended potentiometer position, controlled switch truth, camera source, and capture notes. The annotation tool adds:

- Two ordered rail points defining board orientation.
- Potentiometer center and pointer-tip points.
- Valid, occluded, ambiguous, or rejected status.

The software converts image coordinates to the mathematical angle convention by inverting the image y-axis:

```text
dx = tip_x - center_x
dy = center_y - tip_y
image_pointer_angle = atan2(dy, dx)
board_relative_angle = wrap(image_pointer_angle - board_rotation)
```

Switch truth comes from the controlled capture manifest, not later visual guessing.

## Model

The implemented learned model is a compact native PyTorch CNN trained from randomly initialized weights:

- Shared convolutional encoder.
- Normalized sine/cosine board-orientation head.
- Independent logits for `S_LEFT` and `S_RIGHT`.
- Spatial decoder producing half-resolution Gaussian heatmaps for potentiometer center and pointer tip.
- Early skip connection to preserve small control details.
- Group normalization for consistent behavior during small-batch training and inference.
- A 4-by-4 pooled spatial grid in the switch head, rather than a single whole-image average.
- Spatial KL heatmap loss against normalized Gaussian targets, with subpixel peak decoding.
- Confidence and rejection logic kept outside the raw neural-network output.

No downloaded pretrained backbone is used for the primary result. A pretrained encoder may be evaluated later as a separately reported comparison.

If whole-image resizing removes too much potentiometer detail, a second small CNN may operate on a jittered high-resolution crop from the original photograph. This will be added only if pilot results show that resolution is the problem.

## Evaluation

The original pixelwise heatmap loss was dominated by background pixels. The revised loss makes each keypoint compete over locations; suppressing the entire map cannot improve it. Small-batch BatchNorm produced a training/inference mismatch, and global pooling discarded useful switch-position information. The diagnostic tests these corrections on eight deliberately reused training images before a larger training run. Its pass criteria are engineering checks, not validated deployment thresholds or confidence calibration.

The implemented evaluation command reports:

- Board-orientation angular error.
- Potentiometer center and pointer-tip pixel error.
- Board-relative potentiometer-angle error.
- Per-switch accuracy, precision, recall, and F1.
- Exact two-switch configuration accuracy.
- Counts of defined and undefined potentiometer angles.

Confidence calibration, rejection coverage, error among accepted images, and breakdowns by lighting/framing/difficulty are planned. A held-out session can be selected with `--split validation` or `--split test`, but none has been evaluated yet. Current training-set results are recorded in the [experiment history](cv-experiments.md).

## Integration stages

1. Build reproducible `uv` project metadata, preprocessing, manifests, annotation tooling, and transform tests.
2. Implement classical geometry and color baselines as inspectable references.
3. Implement and smoke-test the custom multi-task PyTorch model.
4. Train on the three-session pilot and diagnose held-out-session failures.
5. Expand the dataset only if the pilot identifies a data-limited result.
6. Calibrate potentiometer angle to resistance and verify switch-to-capacitor mapping.
7. Convert accepted predictions into versioned circuit parameters and run LTspice through a separate controller.
8. Export the frozen model to ONNX and benchmark OpenVINO only after the PyTorch result is established.

Vision error, calibration error, LTspice-model error, and physical measurement error will be reported separately. Simulation output will not be presented as an oscilloscope measurement.

## Local setup

The project requires Python 3.12 and `uv`. Run these PowerShell commands from the repository root, with `uv` available on PATH. Keep all environment state inside ignored repository directories:

```powershell
$repo = (Resolve-Path .).Path
$env:UV_PYTHON_INSTALL_DIR = Join-Path $repo '.uv\python'
$env:UV_CACHE_DIR = Join-Path $repo '.uv\cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $repo '.venv-cv'
uv python install 3.12
uv sync --python 3.12
```

Run commands through `.\.venv-cv\Scripts\python.exe -m oscillator_cv`. Generated datasets, environments, checkpoints, and runs are ignored by Git.

## CLI workflow

Replace `<timestamp>` in the examples with a new run tag, such as `pilot-001`, and `IMAGE` with your photograph path. Quote paths that contain spaces. Use validation results while tuning; reserve the test session for the frozen model.

Generate the three-session pilot plan (120 photos):

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv capture-plan --output data\manifests\pilot-capture-plan.csv
```

For the expanded 240-photo plan, add `--sessions 6` and use a separate output filename. The checked-in `pilot-capture-plan.csv` contains the pilot. The earlier `capture-plan.csv` is retained with its original six-session splits for historical reference; do not use it for the three-session pilot.

Import original JPEGs without modifying them. EXIF orientation is applied first. Images that still display landscape are turned 90 degrees clockwise before aspect-preserving resize and padding to 576×768 RGB. Landscape captures must have the power module on the left; portrait captures must have it at the top. Import and prediction use the same rule, retaining small board-angle variations. Annotation and transform coordinates refer to the canonical portrait image:

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv import-images data\raw --capture-plan data\manifests\pilot-capture-plan.csv --output-dir data\processed\import-<timestamp> --manifest data\processed\manifest-<timestamp>.json
```

Annotate the ordered board axis, potentiometer center/tip, and both switch states:

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv annotate data\processed\manifest-<timestamp>.json --output data\local-annotations\annotations.json
```

In the OpenCV window, click board-axis tail at the buzzer end, axis head at the power-module end, then near the potentiometer to open a magnified crop from the original photograph. In that crop, click the circular dial center and the orange pointer tip. Fractional working-image coordinates are retained. Press `1`/`2` for left OFF/ON, `3`/`4` for right OFF/ON, `v`/`o`/`a`/`r` for valid/occluded/ambiguous/rejected, `n` to save and advance, `z` to reset, `e` to redo the potentiometer points, or `q` to stop. `--review-pot` revisits saved valid labels that have not yet been refined with this zoom, preserving their axis and switch labels; rerunning it resumes unfinished review. `--script` accepts reviewed JSON annotations for reproducible tests and migrations.

Render preprocessing overlays:

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv preprocess-check data\processed\manifest-<timestamp>.json --annotations data\local-annotations\annotations.json --output-dir runs\preprocess-<timestamp>
```

Train, resume into a new run directory, evaluate, and predict:

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv train --manifest data\processed\manifest-<timestamp>.json --annotations data\local-annotations\annotations.json --split train --output-dir runs\train-<timestamp>
.\.venv-cv\Scripts\python.exe -m oscillator_cv train --manifest data\processed\manifest-<timestamp>.json --annotations data\local-annotations\annotations.json --split train --resume runs\train-<timestamp>\checkpoint-epoch-0010.pt --output-dir runs\resume-<timestamp>
.\.venv-cv\Scripts\python.exe -m oscillator_cv evaluate runs\resume-<timestamp>\checkpoint-epoch-0020.pt --manifest data\processed\manifest-<timestamp>.json --annotations data\local-annotations\annotations.json --split validation --output runs\evaluation-<timestamp>.json
.\.venv-cv\Scripts\python.exe -m oscillator_cv predict IMAGE --checkpoint runs\resume-<timestamp>\checkpoint-epoch-0020.pt --output runs\prediction-<timestamp>.json
```

Prediction JSON is deliberately marked `accepted: false` until thresholds are derived from real held-out sessions. Synthetic data is exposed only through explicit `--synthetic` flags for tests and plumbing checks.

`train --seed` defaults to 11 and controls model initialization and minibatch randomness; it also controls fixtures when `--synthetic` is used. Use the same fixture seed for synthetic training and evaluation. New checkpoints record the training seed. Resume restores weights and optimizer state but restarts the seeded random sequence, rather than reproducing an uninterrupted run exactly. Checkpoint loading accepts tensor/dictionary data only and has no unrestricted-pickle fallback; use checkpoints from a trusted source.

### Small-set learning diagnostic

Run from the repository root, using a new output directory:

```powershell
.\.venv-cv\Scripts\python.exe scripts\overfit_check.py --manifest data\processed\s01-manifest.json --annotations data\local-annotations\s01.json --output-dir runs\overfit-<timestamp>
```

This selects P00 and P09, covering eight images and all four switch states. It checks training fit every 25 optimization steps, stops on its diagnostic gate or at 300 steps, and saves checkpoints, per-image predictions, overlays, and a report. It does not modify photographs or labels. `--evaluate-checkpoint` can render a previously saved diagnostic checkpoint without training.

New CLI training runs default to GroupNorm and a 4-by-4 switch pooling grid; `--normalization batch --switch-pool-grid 1` retains the old architecture for comparison. Old checkpoints remain loadable for evaluation, but cannot resume under a different heatmap objective. Diagnostic checkpoints count optimization steps rather than epochs and cannot be resumed by the ordinary training command. Spatial heatmap peaks are not calibrated correctness probabilities.
