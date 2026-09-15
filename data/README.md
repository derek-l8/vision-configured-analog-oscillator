# Image Dataset Protocol

The working image dataset is stored locally and is not committed to ordinary Git history. This directory documents how images are captured, named, labeled, split, and eventually reviewed for publication.

## Capture convention

- Original full-resolution JPEG from the rear 1x iPhone camera.
- Portrait 4:3 image with power module at the top and buzzer at the bottom.
- Complete breadboard and rails visible without clipping.
- Primary dataset captured unpowered with the momentary button released.
- Orange potentiometer pointer and both blue switch tabs unobstructed.
- Normal framing keeps the board at approximately 75-85% of image height.
- In-plane rotation may vary by approximately plus or minus 25 degrees.
- Perspective remains near-overhead, with no more than approximately 15 degrees of deliberate tilt.

The ten-image visibility set includes closer, farther, rotated, tilted, and differently lit examples. It exists to test preprocessing and control visibility; it is not training or evaluation data.

## Control names

- `S_LEFT` and `S_RIGHT` refer to the physical left and right switches in the portrait reference view.
- `1` means the actuator points toward the power module and the switch is ON.
- `0` means the actuator points toward the buzzer and the switch is OFF.
- `P00` through `P09` are requested visual positions, not measured resistance values.

The ten approximate potentiometer targets are:

```text
P00  30 degrees    P05  163 degrees
P01  57 degrees    P06  190 degrees
P02  83 degrees    P07  217 degrees
P03 110 degrees    P08  243 degrees
P04 137 degrees    P09  270 degrees
```

Actual pointer angle is calculated from annotation. Later physical calibration maps that angle to resistance.

## Full-session grid

Each full session contains 40 photographs: ten potentiometer positions by four switch configurations. At each potentiometer position, the efficient switch order is:

```text
L0R0 -> L1R0 -> L1R1 -> L0R1
```

A generated capture manifest assigns approximate board-rotation bins independently of the control state. Exact board rotation is derived from annotation rather than treated as known from the capture request.

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
  sample-images/      # small, reviewed public subset added later
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

## Splitting and publication

- Split by complete capture session, never by randomly mixing photographs from one setup.
- Keep augmented images in the same split as their source.
- Freeze at least one unseen session before final model selection.
- Keep an external-camera challenge set separate from primary model selection.
- Do not report the ten visibility photographs as training data or model evaluation.
- Remove location metadata and review backgrounds before publishing any original image.
- Publish only reviewed sample images in Git; distribute any complete dataset separately if needed.
