# Computer-Vision Implementation Plan

## Objective

Given a saved photograph of this fixed breadboard, estimate its visible persistent controls and convert those estimates into parameters for the matching LTspice circuit. The task is intentionally limited to one known physical build. It is not general breadboard recognition.

Current evidence is limited to a completed ten-image visibility check. The model, controller, and calibration have not yet been implemented or evaluated.

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

The electrical mapping from `S_LEFT` and `S_RIGHT` to LTspice parameters `Cadd1` and `Cadd2` will be established separately with a powered-off continuity test. It will not be inferred from image appearance.

## Data design

The physical layout, orange potentiometer pointer, and blue switch markers remain fixed after data collection begins. The primary dataset is photographed unpowered with the momentary button released.

Collection proceeds in stages:

| Stage | Images | Purpose |
|---|---:|---|
| Visibility | 10 | Inspect control detail after exact preprocessing; never used for training metrics |
| Smoke | 40-60 | Prove image loading, annotations, transforms, losses, and the training loop |
| Pilot | 120 | Three independent 40-image capture sessions |
| Expanded | 240 | Six 40-image sessions, only if the pilot justifies expansion |

Each full session covers ten approximately even potentiometer positions across the usable 30-270 degree sweep and all four switch combinations. The exact angle is derived from annotated center and pointer-tip coordinates rather than assumed from the requested capture position.

Capture sessions vary lighting, background, distance, and perspective. Board rotation also varies within sessions. Dataset splits use complete sessions so near-duplicate photographs from one setup cannot appear in both training and evaluation.

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

The first learned model is a compact native PyTorch CNN trained from randomly initialized weights:

- Shared convolutional encoder.
- Normalized sine/cosine board-orientation head.
- Independent logits for `S_LEFT` and `S_RIGHT`.
- Spatial decoder producing half-resolution Gaussian heatmaps for potentiometer center and pointer tip.
- Early skip connection to preserve small control details.
- Confidence and rejection logic kept outside the raw neural-network output.

No downloaded pretrained backbone is used for the primary result. A pretrained encoder may be evaluated later as a separately reported comparison.

If whole-image resizing removes too much potentiometer detail, a second small CNN may operate on a jittered high-resolution crop from the original photograph. This will be added only if pilot results show that resolution is the problem.

## Evaluation

Evaluation reports separate metrics for:

- Board-orientation angular error.
- Potentiometer center and pointer-tip pixel error.
- Board-relative potentiometer-angle error.
- Per-switch accuracy, precision, recall, and F1.
- Exact two-switch configuration accuracy.
- Confidence, rejection coverage, and error among accepted images.
- Performance by held-out capture session, lighting, framing, and difficulty.

Performance targets will be set after the smoke dataset exposes the practical label noise and baseline difficulty. No accuracy is claimed before evaluation on a held-out capture session.

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
