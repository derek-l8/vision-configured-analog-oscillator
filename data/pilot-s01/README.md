# First annotated capture session (s01)

This package contains the 40 processed photographs and manual labels used for the
first real-photo CNN experiments. All images belong to the **training split**.
There is no held-out validation or test data in this package.

## Contents

- `images/`: 40 RGB JPEGs, each 576 × 768 pixels.
- `manifest.json`: image identities, capture targets, switch states, and paths.
- `annotations.json`: manual board-axis, potentiometer-center, and pointer-tip labels.

The session covers ten requested pot positions (45°–315°, every 30°), each with
all four switch configurations. Requested angles are approximate capture targets,
not measured dial positions or resistance. Actual labeled angle is derived from
the center/tip points and board axis.

Labels use working-image pixels: x increases right, y increases down. The board
axis runs from `axis_tail` near the buzzer to `axis_head` near the power module.
`S_LEFT` and `S_RIGHT` name the switches in the portrait view; 1 means ON
(actuator toward the power module), and 0 means OFF.

## Image preparation and privacy

The working images were normalized to portrait and resized before training.
These public copies preserve their decoded RGB pixels exactly; JPEG application
and comment segments, including EXIF, XMP, IPTC, and ICC metadata, were removed
without recompression. Full-resolution originals remain private and unchanged.
The image backgrounds were also visually reviewed.

The original manual labels are retained, including their annotation-method field.
They were refined using original-resolution zoom and expressed in working-image
coordinates. Full-resolution zoom is not available from this public package.

Both manifest paths point to the published processed image, relative to the
manifest's directory. Its transform is therefore identity, not the earlier
full-resolution-to-working resize. Moving the whole package preserves its paths.

## Load and train

After the environment setup in the [implementation guide](../../docs/computer-vision-plan.md),
run from the repository root, using a new output directory:

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv train --manifest data\pilot-s01\manifest.json --annotations data\pilot-s01\annotations.json --split train --epochs 10 --batch-size 4 --device cpu --output-dir runs\public-pilot-001
```

This starts a new run; it does not reproduce the final 60-epoch result by itself.
The [experiment history](../../docs/cv-experiments.md) records training budgets
and the lower-learning-rate resume. Checkpoints and generated run outputs are not
included in this package.

## Limits

These photos come from one setup, with similar lighting/background and a narrow
board-rotation range. Results on them measure fit to the supplied labels, not
generalization, calibrated confidence, or physical measurement accuracy.
Independent sessions and angle-to-resistance calibration are still needed.
