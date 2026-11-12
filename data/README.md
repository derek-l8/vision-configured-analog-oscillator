# Image Dataset Protocol

The private working dataset remains local. The [published first session](pilot-s01/README.md) contains its 40 processed training photos, manual annotations, and a portable manifest. This directory documents capture, labels, splits, and publication.

## Capture convention

- Original full-resolution JPEG from the rear 1x iPhone camera.
- Portrait 4:3 image with power module at the top and buzzer at the bottom.
- Landscape files with power on the left are also accepted: after EXIF correction, import and prediction turn them clockwise into portrait before resizing. Originals remain unchanged.
- Complete breadboard and rails visible without clipping.
- Primary dataset captured unpowered with the momentary button released.
- Orange potentiometer pointer and both blue switch tabs unobstructed.
- Normal framing keeps the board at approximately 75-85% of image height.
- Keep the board roughly upright. Small alignment differences from naturally repositioning the camera are expected; no deliberate board rotation or angle cycle is required.
- Stay near-overhead. Small natural perspective differences are expected; avoid deliberate tilt and keep any incidental tilt modest (roughly within 15 degrees).
- Keep wires, components, and colored markers fixed after collection starts.

The ten-image visibility set includes closer, farther, rotated, tilted, and differently lit examples. It exists to test preprocessing and control visibility; it is not training or evaluation data.

## Control names

- `S_LEFT` and `S_RIGHT` refer to the physical left and right switches in the portrait reference view.
- `1` means the actuator points toward the power module and the switch is ON.
- `0` means the actuator points toward the buzzer and the switch is OFF.
- `P00` through `P09` are requested visual positions, not measured resistance values.

The ten approximate potentiometer targets are:

```text
P00  45 degrees    P05 195 degrees
P01  75 degrees    P06 225 degrees
P02 105 degrees    P07 255 degrees
P03 135 degrees    P08 285 degrees
P04 165 degrees    P09 315 degrees
```

Actual pointer angle is calculated from annotation. Later physical calibration maps that angle to resistance.

## Full-session grid

Each full session contains 40 photographs: ten potentiometer positions by four switch configurations. At each potentiometer position, the efficient switch order is:

```text
L0R0 -> L1R0 -> L1R1 -> L0R1
```

The generated capture manifest uses a nominal rotation intent of 0 degrees to mean roughly aligned, not exact alignment or a measured angle. Reposition the camera naturally without assigning an orientation to a switch state. Actual board rotation is derived from annotation.

## Next capture: s02

The 40-photo training session `s01` is complete. Capture `s02` as a fresh validation session using the same 10-by-4 control grid. Set up the camera again and make an ordinary change to lighting direction or background; keep the board roughly aligned and normally framed. Evaluate the existing checkpoint on `s02` before retraining.

| Pot ID | Approximate pointer angle | s02 photo numbers |
|---|---:|---|
| P00 | 45 degrees | 041-044 |
| P01 | 75 degrees | 045-048 |
| P02 | 105 degrees | 049-052 |
| P03 | 135 degrees | 053-056 |
| P04 | 165 degrees | 057-060 |
| P05 | 195 degrees | 061-064 |
| P06 | 225 degrees | 065-068 |
| P07 | 255 degrees | 069-072 |
| P08 | 285 degrees | 073-076 |
| P09 | 315 degrees | 077-080 |

The angles only need to be approximate; labels use the annotated center, pointer tip, and board axis. Use the filenames in `manifests/pilot-capture-plan.csv`, beginning with `s02_p00_l0_r0_041.jpg` and ending with `s02_p09_l0_r1_080.jpg`. Numbering continues across sessions.

Capture `s03` as another independent session under ordinary intended-use conditions and reserve it for the final frozen-model test. The initial pilot totals 120 photos including the existing `s01`. Expand only if validation shows that additional training data is needed.

Recommended filenames use lowercase identifiers:

```text
s01_p00_l0_r0_001.jpg
s01_p00_l1_r0_002.jpg
s01_p00_l1_r1_003.jpg
s01_p00_l0_r1_004.jpg
```

## Local layout

```text
data/
  README.md
  manifests/          # reviewed capture manifests and split definitions
  pilot-s01/          # reviewed public training photos, labels, and manifest
  raw/                # original private JPEGs; ignored by Git
  processed/          # generated tensors/previews; ignored by Git
  local-annotations/  # working annotations; ignored by Git
```

The initial capture manifest is tabular and records:

```text
image_id
relative_path
session_id
build_revision
pot_target_id
s_left_on
s_right_on
capture_rotation_intent_deg
camera_source
capture_notes
```

The annotation output adds ordered rail points, potentiometer center, potentiometer tip, validity status, and optional quality notes. A version field is required in both formats so later schema changes can be migrated deliberately.

The board axis is ordered from the buzzer/bottom end to the power-module/top end. Potentiometer center-to-tip follows the mathematical convention: right is 0 degrees, up is 90 degrees, and angles increase counterclockwise. The tip is the pointed end of the orange tape. For both switches, blue toward the board top is ON and blue toward the bottom is OFF.

## Splitting and publication

The default `pilot-capture-plan.csv` contains 120 photos: `s01` is training, `s02` validation, and `s03` test. `capture-plan --sessions 6` adds training sessions `s04`–`s06` while keeping the same held-out sessions. The earlier `capture-plan.csv` is retained for historical reference, with its original six-session splits and deliberate rotation targets; do not use it for new captures. Existing imported manifests retain their original splits and capture metadata; regenerating a capture plan does not change historical datasets.

- Split by complete capture session, never by randomly mixing photographs from one setup.
- Keep augmented images in the same split as their source.
- Freeze at least one unseen session before final model selection.
- Keep an external-camera challenge set separate from primary model selection.
- Do not report the ten visibility photographs as training data or model evaluation.
- Remove location metadata and review backgrounds before publishing any original image.
- The reviewed 40-photo processed session is included in Git (about 7.4 MB of JPEGs). Full-resolution originals and private working directories remain ignored; review size and privacy before publishing further sessions.
