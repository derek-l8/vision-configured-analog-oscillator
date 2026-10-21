# Inspecting and rerunning the project

This page is for readers who want to examine the circuit model, data, or code
behind the demonstrations. Watching the videos and reading the results requires
no installation.

## LTspice reference run

Open [analog-oscillator-v8.asc](../ltspice/analog-oscillator-v8.asc) in LTspice
(the reference used version 26.0.2) and select **Run**. The saved directives set
`Rp = 5.1 kΩ`, open timing switches as `Cadd1 = Cadd2 = 1 pF`, and a 12-second
transient with a simulated button hold from 6–7 seconds.

Probe `O1` and `FILT` for the raw and filtered waveforms, and `ENV` for the
envelope. Current through `RBUZ` is the modeled buzzer current. Open the
**SPICE Error Log** to inspect the `.meas` results and compare them with
[simulation-results.txt](../ltspice/simulation-results.txt).
The [circuit notes](circuit-design.md#interpreting-the-simulation) describe model assumptions.

## Computer-vision evidence

The [s01 package](../data/pilot-s01/README.md) includes 40 processed images,
manual labels, and a relative-path manifest. All images are training data.
The [experiment history](cv-experiments.md) records the reported 60-epoch result;
the corresponding trained weights and generated reports have not been published.
That checkpoint cannot currently be downloaded and re-evaluated from this repository.

The following steps inspect the included data and optionally train a new model.
An independent capture session is needed to assess performance on new photographs.

### Environment

Use Windows PowerShell and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Download and extract this repository, or clone it with Git. Run the commands from
the repository root, where `pyproject.toml` is located. `uv` manages Python 3.12
and the locked dependencies, including PyTorch and OpenCV:

```powershell
$repo = (Resolve-Path .).Path
if (-not (Test-Path .\pyproject.toml)) { throw 'Run from the repository root.' }
$env:UV_PYTHON_INSTALL_DIR = Join-Path $repo '.uv\python'
$env:UV_CACHE_DIR = Join-Path $repo '.uv\cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $repo '.venv-cv'
uv python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
uv sync --locked --python 3.12
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
.\.venv-cv\Scripts\python.exe -m oscillator_cv --help
```

These examples use CPU and require no downloaded model. Choose a new run name
each time; training refuses an existing output directory. Generated files under
`runs/` and environment directories stay ignored by Git.

### Inspect the published labels

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv preprocess-check data\pilot-s01\manifest.json --annotations data\pilot-s01\annotations.json --output-dir runs\public-overlays-001
```

Open the generated images to inspect the board axis, center, and pointer-tip
labels. Full-resolution annotation zoom is unavailable because the original
photographs are not part of the public dataset.

### Optional training and evaluation

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv train --manifest data\pilot-s01\manifest.json --annotations data\pilot-s01\annotations.json --split train --epochs 10 --batch-size 4 --seed 11 --device cpu --output-dir runs\public-pilot-001
if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
.\.venv-cv\Scripts\python.exe -m oscillator_cv evaluate runs\public-pilot-001\checkpoint-epoch-0010.pt --manifest data\pilot-s01\manifest.json --annotations data\pilot-s01\annotations.json --split train --device cpu --output runs\public-pilot-001\training-evaluation.json
```

This produces a new ten-epoch checkpoint and training-set metrics; it does not
reproduce the recorded 60-epoch result. The existing package has no validation
or test split. Requesting either split will fail until additional sessions exist.

To inspect one prediction from the new checkpoint:

```powershell
.\.venv-cv\Scripts\python.exe -m oscillator_cv predict data\pilot-s01\images\s01_p00_l0_r0_001.jpg --checkpoint runs\public-pilot-001\checkpoint-epoch-0010.pt --device cpu --output runs\public-pilot-001\prediction.json
```

Prediction JSON reports geometry and switch estimates. `accepted` remains `false`
until held-out evaluation establishes confidence thresholds; predictions do not
yet control LTspice.

For new capture sessions, annotation, architecture, or the small-set diagnostic,
see [architecture and development](computer-vision-plan.md).
