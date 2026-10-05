# CV experiment history

This records the training iterations, including unsuccessful and interrupted runs.
The implementation and experiments were developed with Codex assistance.

All real-photo results below evaluate the **same images used for training**.
They show whether the model can fit the supplied labels, not accuracy on new
photographs. Pixel and angle errors are relative to manual annotations, not
independently measured physical dial positions. Predictions remain unaccepted.

## Initial execution check — September 15, 2026

`unattended-20260915-005542`: trained on 12 synthetic fixtures for one epoch,
then resumed for one additional epoch in a separate output folder. Evaluation
and structured prediction completed on CPU. Fourteen tests passed at this stage.

The two-epoch checkpoint scored 25% exact switch accuracy, 17.17 px center error,
18.63 px tip error, and 47.90° pot-angle MAE. These artificial fixtures and short
runs checked execution, checkpoint resume, and output format—not model quality
on the breadboard. The ten real photographs were reviewed for visibility only.

Evidence: `runs/unattended-20260915-005542/REPORT.md`,
`smoke/train-20260915-011701/checkpoint-epoch-0001.pt`,
`smoke/resume-20260915-011701/checkpoint-epoch-0002.pt`, and
`smoke/evaluation-final.json` within that run folder.

## First real-photo iterations — October 4, 2026

Session `s01` contains 40 photos: ten approximate pot positions, each with all
four switch configurations. Before training, mixed iPhone image orientations
were normalized to portrait with power at the top; an original-resolution zoom
was added to make center/tip annotation practical. Original photographs were
retained.

Common setup: 576 × 768 RGB input, custom CNN trained from scratch, CPU, seed 11,
learning rate 0.001, and batch size 4 unless noted. The small-set diagnostics use
eight photos from P00 and P09, covering all four switch configurations at each.
The annotations were unchanged across these diagnostics and the final review.

“Exact switches” requires both states to be correct. Center/tip columns are mean
Euclidean errors in working-image pixels; angle columns are mean absolute errors.

| Run | Images / budget | Exact switches | Center / tip (px) | Pot angle | Board angle |
|---|---|---:|---:|---:|---:|
| R1: original pilot | 40 / 10 epochs | 27.5% | 243.37 / 298.28 | 72.43° | 2.75° |
| R2: spatial-loss diagnostic, interrupted | 8 / 50 steps | 25% | 2.18 / 6.30 | Not retained in notes | 34.07° |
| R3: GroupNorm diagnostic, interrupted | 8 / 125 steps | 25% | 0.40 / 0.41 | 1.14° | 1.17° |
| R4: spatial switch-head diagnostic | 8 / 150 steps | 100% | 1.09 / 1.68 | 5.65° | 1.27° |
| R5: corrected full pilot | 40 / 40 epochs | 87.5% | 1.14 / 1.29 | 5.81° | 1.02° |
| R6: lower-rate resume | 40 / 20 more epochs, 60 total | 100% | 0.34 / 0.33 | 1.44° | 0.89° |

### R1 — Establish the real-photo baseline

Run: `s01-pilot-001`; checkpoint: `checkpoint-epoch-0010.pt`.
Used pixelwise heatmap BCE, BatchNorm, global-average switch pooling, and integer
heatmap peak decoding. Both switches scored 52.5% individually; pot predictions
were far from their labels. Falling training loss did not establish useful fit.
The large blank background in the heatmap targets motivated a loss revision.

Evidence: `runs/s01-pilot-001/training-evaluation.json`.

### R2 — Change the heatmap objective

Run: `s01-overfit-spatial-kl-001`; batch size 8.
Replaced pixelwise BCE with spatial KL divergence against normalized Gaussian
targets, so matching the point location—not mostly empty background—drives the
heatmap objective. Localization improved, but inference-mode orientation error
was 34.07° despite low training-mode orientation loss. Interrupted after the
step-50 check to investigate that mismatch; no completed checkpoint was saved.

Evidence: `runs/s01-overfit-spatial-kl-001/NOTES.md` (recorded console check).

### R3 — Remove batch-statistics dependence

Run: `s01-overfit-spatial-kl-group-001`; checkpoint: `checkpoint-step-0125.pt`.
Changed to GroupNorm and batch size 4, and used subpixel heatmap-peak refinement.
Center/tip fit became subpixel and pot-angle MAE reached 1.14°, but exact switch
accuracy stayed at 25%. Stopped after 125 steps to revise the switch head rather
than continue this configuration. The diagnostic gate did not pass.

Evidence: `runs/s01-overfit-group-review/report.json`, an evaluation of this
checkpoint, not another training run.

### R4 — Preserve switch-position information

Run: `s01-overfit-spatial-kl-grid-001`; checkpoint: `checkpoint-final.pt`.
Replaced whole-image average pooling in the switch head with a 4 × 4 spatial
grid. The eight-photo training-fit gate passed at step 150, ending the run early.
The gate requires all switches correct, board-angle MAE below 2°, center/tip
mean errors below 3 px, pot-angle MAE below 15°, and all pot angles defined.

Pot-angle error was higher than R3's despite better switch fit. This was a joint
task-fit check, not a claim that every metric improved or that the model is ready
for deployment.

Evidence: `runs/s01-overfit-spatial-kl-grid-001/report.json` and prediction overlays.

### R5 — Train the corrected model on all 40 photos

Run: `s01-pilot-spatial-kl-001`; trained from scratch for 40 epochs with spatial
KL, GroupNorm, and the 4 × 4 switch head. Saved checkpoints show the trajectory:

| Checkpoint | Exact switches | Center / tip (px) | Pot-angle MAE |
|---|---:|---:|---:|
| Epoch 10 | 25% | 2.19 / 7.00 | 37.30° |
| Epoch 20 | 25% | 1.61 / 1.43 | 4.70° |
| Epoch 40 | 87.5% | 1.14 / 1.29 | 5.81° |

These are checkpoints of one training run, not separate experiments. At epoch
40 the right switch was correct on all photos; five left-switch errors remained.
Angle error was not monotonic as switch fit improved.

Evidence: `runs/s01-pilot-spatial-kl-001/history.json` and
`training-evaluation-0010.json`, `-0020.json`, and `-0040.json` in that folder.

### R6 — Refine from epoch 40 at a lower learning rate

Run: `s01-pilot-spatial-kl-refine-001`; resumed R5 for 20 more epochs at 0.0003.
Final checkpoint: `checkpoint-epoch-0060.pt`. All 40 switch configurations were
correct; all pot angles were defined. Maximum center and tip errors were each
0.92 px, and maximum pot-angle error was 3.55°. Final overlays were inspected,
original-JPEG prediction executed, and 29 regression/integration tests passed.

Evidence: `runs/s01-pilot-spatial-kl-refine-001/training-evaluation.json` and
`runs/s01-pilot-final-review/REPORT.md`, `report.json`, and contact sheets.
The final-review folder records evaluation, not additional training.

## What these results do and do not establish

The corrected implementation fits this session's labels. The baseline-to-final
comparison combines loss, normalization, pooling, decoding, and training-budget
changes; it does not isolate any one change's causal effect.

Board rotations only span about −3.59° to +0.93°. An always-upright prediction
already scores 1.12° MAE, so the final 0.89° result is a weak orientation test.
Subpixel label error also does not imply subpixel physical measurement accuracy.
An independent capture session has not been evaluated. Confidence calibration,
angle-to-resistance calibration, physical frequency measurements, and the
automated LTspice controller remain unvalidated.

Original photos, generated run artifacts, and checkpoints remain local and
ignored by Git. The [published s01 package](../data/pilot-s01/README.md) contains
the 40 processed training inputs and unchanged manual labels. These iterations used uncommitted
working changes based on `d991f53`, not separately committed source snapshots.
Checkpoints retain model configuration, but the intermediate source states are
not all preserved as Git commits.

## Recording subsequent runs

Publication-review fixes after R6 added the three-session pilot split, wired the CLI training seed through to the training loop, and restricted checkpoint loading. These changes did not retrain a model or alter the R1–R6 results, original data, annotations, or imported manifests. New ordinary CLI runs default to seed 11; the earlier synthetic check used fixture seed 7 and training seed 11.

Validation after these fixes: 36 tests passed; the older synthetic, baseline, diagnostic, and final checkpoints loaded with the restricted loader, and the final optimizer state loaded for resume. The existing manifest, annotation file, and final checkpoint hashes were unchanged.

Dataset publication removed JPEG application/comment metadata without recompression and added a relative-path manifest for the processed inputs. The published inputs and labels are unchanged; publication is not another training run or a held-out evaluation.

Publication validation: 39 tests passed, including loading the package after relocation. All 40 public inputs produced tensors and training targets identical to the private dataset; labels were identical, and hashes for the original photos, working images, private manifest/labels, and final checkpoint were unchanged. Preprocessing overlays rendered for all 40 public images, and the image backgrounds were visually reviewed.

Append an entry after each training experiment, including failed or interrupted
ones. Evaluate checkpoints under their parent run; do not count re-evaluation
as new training. Retain earlier results and label later corrections explicitly.

Each entry should record:

- Run ID, date, source commit or uncommitted status, and parent checkpoint if resumed.
- Question being tested and changes from the comparison run.
- Dataset/session IDs, train/validation/test split, preprocessing, and label version.
- Seed, device, batch size, learning rate, epochs or optimization steps.
- Metrics and their evaluation scope, stopping reason, and observed regressions.
- Interpretation, resulting decision, and paths to reports/checkpoints/overlays.
