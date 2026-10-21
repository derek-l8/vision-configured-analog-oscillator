# Computer vision: architecture and development

For inspecting the published dataset or running its examples, start with
[verification](verification.md). This page covers the model and ongoing data collection.

## Objective

Given a saved photograph of this fixed breadboard, estimate its visible persistent controls and convert those estimates into parameters for the matching LTspice circuit. The task is intentionally limited to one known physical build. It is not general breadboard recognition.

The ten-image visibility check, synthetic execution tests, and first 40-photo real-image training run are complete. A corrected model passed an eight-photo overfit diagnostic covering all four switch combinations and two potentiometer positions. Evaluation has used training photographs only; no held-out capture session has been tested. The LTspice controller and physical calibration are not implemented.

## Current visual conventions

- Primary source: original iPhone JPEG photographs.
- Canonical input: portrait, with the power module at the top and buzzer at the bottom.
- Model size: `576 x 768`, using aspect-preserving resize and padding when required.
- Expected board alignment: roughly upright, with small natural camera alignment differences; deliberate rotation across a fixed angular range is outside the current capture scope.
- Expected camera perspective: near-overhead, with small natural perspective differences; avoid deliberate tilt and keep incidental tilt modest (roughly within 15 degrees).
- `S_LEFT`: maintained switch on the left in the canonical portrait view.
- `S_RIGHT`: maintained switch on the right in the canonical portrait view.
- Switch actuator toward the power module: ON.
- Switch actuator toward the buzzer: OFF.
- Potentiometer angle: 0 degrees right, 90 degrees up, increasing counterclockwise.
- Potentiometer pointer tip: the pointed end of the orange tape arrow.

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

Capture sessions use fresh camera setups and ordinary changes in lighting, background, or framing within the intended near-overhead conditions. Keep the board roughly aligned; repositioning the camera naturally supplies small alignment differences without a deliberate rotation pattern. Dataset splits use complete sessions so near-duplicate photographs from one setup cannot appear in both training and evaluation.

Next, capture and annotate `s02` and evaluate the existing 60-epoch checkpoint unchanged before retraining. Use validation failures to decide whether additional training captures or model changes are necessary. Reserve an independent `s03` for the final frozen-model test. See the [s02 capture checklist](../data/README.md#next-capture-s02) for the grid and filenames. Generated plans record 0 degrees as the nominal alignment intent; measured rotation still comes from annotation.

The three-session pilot uses `s01` for training, `s02` for validation, and `s03` for the final test. The six-session plan keeps `s02` and `s03` held out and adds `s04`–`s06` to training. Choose the split before import; generating a new capture plan does not change an existing imported manifest. The earlier 40-photo runs used only `s01`.

The [published s01 package](../data/pilot-s01/README.md) contains all 40 processed inputs and their manual labels. Its image paths resolve relative to the manifest, so it can be moved or loaded from another working directory. The [verification examples](verification.md#optional-training-and-evaluation) use this package without importing or annotating again. All records are training data; validation/test commands require additional sessions.

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

## Collecting another session

Follow the [capture protocol](../data/README.md), including the `s02` grid and
filenames. Demo-video rotation and cropping are presentation edits; training
photos still use the portrait convention with power at the top.
The checked-in `pilot-capture-plan.csv` specifies roughly upright alignment.
The older `capture-plan.csv` retains historical deliberate-rotation targets and
must not be used for new captures.

After the [environment setup](verification.md#environment), place your original
JPEGs in `data/raw/` using the plan's filenames. Run these PowerShell commands
from the repository root, choosing new output names for each import:

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv import-images data\raw --capture-plan data\manifests\pilot-capture-plan.csv --output-dir data\processed\session-001 --manifest data\processed\session-001.json
if ($LASTEXITCODE -ne 0) { throw 'Image import failed.' }
.\.venv-cv\Scripts\python.exe -m oscillator_cv annotate data\processed\session-001.json --output data\local-annotations\session-001.json
.\.venv-cv\Scripts\python.exe -m oscillator_cv preprocess-check data\processed\session-001.json --annotations data\local-annotations\session-001.json --output-dir runs\session-overlays-001
```

Import applies EXIF orientation first. Remaining landscape images rotate 90°
clockwise, then resize and pad to 576 × 768. Landscape originals must have power
on the left; portrait originals must have it at the top. Originals stay unchanged.

In the annotation window, click the board-axis tail at the buzzer end, then the
head at the power end. Click near the potentiometer to open original-resolution
zoom, then select its center and the orange arrow tip.

| Key | Action |
|---|---|
| `1` / `2` | Left switch OFF / ON |
| `3` / `4` | Right switch OFF / ON |
| `v` / `o` / `a` / `r` | Valid / occluded / ambiguous / rejected |
| `n` / `q` | Save and advance / stop |
| `z` / `e` | Reset / redo potentiometer points |

`--review-pot` refines saved valid labels with original-resolution zoom while
preserving axis and switch labels. Review overlays before training or evaluation.
Keep imported manifests and labels local; publishing another session requires a
separate image and metadata review.

The command below generates an expanded plan in an ignored working directory.
It adds training sessions `s04`–`s06` while retaining `s02` for validation and
`s03` for the final test:

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv capture-plan --sessions 6 --output data\local-annotations\expanded-plan.csv
```

Use its path with `import-images --capture-plan` for expanded captures. Generating
a plan does not alter earlier imports or their splits. Once `s02` is annotated,
evaluate the existing checkpoint unchanged before deciding on more training.

## Training and diagnostics

[Verification](verification.md#optional-training-and-evaluation) contains training,
evaluation, and prediction commands using the published data. For new sessions,
substitute the manifest and annotation paths produced by your import. Select
`--split validation` for `s02`; keep `s03` frozen for final evaluation.

To run the eight-image learning diagnostic on the included data:

```powershell
.\.venv-cv\Scripts\python.exe scripts\overfit_check.py --manifest data\pilot-s01\manifest.json --annotations data\pilot-s01\annotations.json --output-dir runs\public-overfit-001
```

It selects P00 and P09 with all four switch states, checks fit every 25 steps,
and stops on its diagnostic gate or at 300 steps. It saves predictions, overlays,
checkpoints, and a report. `--evaluate-checkpoint` reviews a saved diagnostic
checkpoint without training. Diagnostic checkpoints count steps and cannot be
resumed by ordinary epoch-based training.

New CLI runs use GroupNorm, a 4 × 4 switch pooling grid, and seed 11. The seed
controls initialization and minibatch randomness; use the same fixture seed
when comparing synthetic runs. `--normalization batch --switch-pool-grid 1`
selects the earlier architecture for comparison. Synthetic fixtures require an
explicit `--synthetic` flag.

Ordinary `train --resume` restores model and optimizer state and runs `--epochs`
additional epochs into a new output directory. It restarts the seeded random
sequence, so it does not exactly reproduce uninterrupted training. Checkpoints
with another heatmap objective can be evaluated but cannot resume this loss.
Only tensor/dictionary checkpoints are accepted; use files from a trusted source.
Spatial heatmap peaks are not calibrated correctness probabilities.
