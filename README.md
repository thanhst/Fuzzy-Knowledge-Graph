# Deep-learning baselines for FKG-MM

This is the compact reviewer branch for the four baselines used in the
patient-level diabetic-retinopathy comparison.

## Models

- `mlp`: tabular MLP;
- `resnet`: image-only ResNet-50;
- `early_fusion`: ResNet-50 image embedding concatenated with an MLP tabular embedding;
- `late_fusion`: learned weighted ensemble of ResNet-50 and MLP logits.

All models use the same outer train/test split and five patient-grouped
validation folds. A patient never occurs on both sides of a split.

## Included data

The branch includes the complete tabular table, labels, image/patient IDs,
outer split, five inner folds, and the latest reported result table. The raw
fundus JPEGs are from the controlled-access BRSET distribution and are not
republished in this public Git branch.

After obtaining BRSET through its official data-use process, create the exact
1,529-image compact set used by these manifests:

```bash
python tools/prepare_images.py /path/to/BRSET/fundus_photos
```

The command checks that every required image is present and writes 224x224
JPEGs to `data/fundus_photos_224/`. The fixed manifests already point there.

## Run

Python 3.10+ is recommended:

```bash
git clone --branch baseline --single-branch https://github.com/thanhst/Fuzzy-Knowledge-Graph.git
cd Fuzzy-Knowledge-Graph
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --validate-data-only
python run.py --models all --resnet-arch resnet50 --epochs 10 --batch-size 16 --results-dir outputs/baseline_5fold
```

On Windows, `run.bat` validates inputs and runs all four models. `run_mlp.bat`
runs the tabular baseline immediately from the bundled data without images.
Use `--device cuda` on a CUDA machine; `auto` is the default. Add
`--run-final-test` only after model selection if an outer-test estimate is
required.

For a quick code-path check after preparing images:

```bash
python run.py --models all --folds 1 --epochs 1 --max-train-batches 1 --max-eval-batches 1 --device cpu --results-dir outputs/smoke
```

The runner writes fold predictions, metrics, mean/sample-standard-deviation
summary, timings, configuration, and a console log. The full five-fold values
reported in the experiment are in `results/reference_baselines.csv`; a new
training run may vary slightly with hardware and PyTorch/CUDA versions.
