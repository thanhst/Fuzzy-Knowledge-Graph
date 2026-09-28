# FKG-MM reproducible baseline

This is a small, standalone reviewer package for the proposed multimodal Fuzzy
Knowledge Graph (FKG-MM). It reproduces the latest reported native FKG-MM row
from the patient-aware five-fold experiment dated 2026-09-23.

## Reproduce

Python 3.10 or newer is recommended.

```bash
git clone --branch baseline --single-branch https://github.com/thanhst/Fuzzy-Knowledge-Graph.git
cd Fuzzy-Knowledge-Graph
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --check-reference
```

On Windows, after installing the dependency, `run.bat` is the one-click entry
point. A successful run prints:

```text
accuracy       91.7 +/-  0.3
f1             16.6 +/- 12.9
auc_roc        80.7 +/-  6.6
specificity    98.7 +/-  0.9
sensitivity    11.4 +/- 10.2
reference labels match: True
```

Machine-readable outputs are written to `outputs/latest/`.

## What is included

- `fkg_mm.py`: compact NumPy implementation of the published native FKG
  training and inference equations.
- `run.py`: patient-leakage checks, five-fold evaluation, metrics, and exact
  comparison against archived native predictions.
- `data/fold_01` ... `data/fold_05`: the exact multimodal fuzzy-rule inputs
  consumed by FKG-MM, selected-feature maps, and patient/image IDs.
- `results/reference_predictions`: predictions produced by the native runner.
- `results/baseline_table.csv`: concise paper-facing comparison for the four
  conventional baselines, two unimodal FKG controls, and proposed FKG-MM.
- `results/latest_baseline_comparison.csv`: the complete latest comparison
  table (MLP, ResNet-50, early/late fusion, FKG-UM, and FKG-MM variants).

The split contains 1,208 outer-train images from 729 patients. K-fold is
performed only inside this outer-train set. Each validation patient appears in
exactly one fold, and every fold has zero train/validation patient overlap.
Training FRB tables contain about 1,778 rows because SMOTE was applied only to
the training part of each fold.

## Scope and evidence boundary

This package starts from exported FIS rule tables, which are the direct input
to FKG-MM. It does not include the original fundus images or rerun image feature
extraction, FIS clustering, MLP, ResNet-50, or early/late fusion training. Their
latest reported numbers are retained only in the comparison CSV. Therefore,
`python run.py --check-reference` independently verifies the FKG-MM stage and
its metrics; it is not an end-to-end reproduction from raw images.

Data provenance: `frb_patient_id_20260923`, source experiment
`KFold_feature_selection_rerun_20260921`, patient-grouped split with
`patient_overlap_count=0`. Metrics use the diabetic-retinopathy class as the
positive class and sample standard deviation across five folds (`ddof=1`).
