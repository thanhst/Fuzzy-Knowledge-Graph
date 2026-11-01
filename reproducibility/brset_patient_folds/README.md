# BRSET patient folds: code and verified reproduction

This folder publishes the verified **patient ID splitting** workflow. Run it from
the root of this repository. The runner calls the existing source functions;
it does not copy or silently replace either algorithm.

| `--method` | Original function | Meaning |
|---|---|---|
| `custom` (default) | `assign_train_folds()` in [create_root_data_image_train_test_split.py](../../Source_code/main/diabetic_retinopathy/create_root_data_image_train_test_split.py) | Custom allocation balancing patient counts, image counts and image-label counts. Reproduces the existing 1,208-image folds. |
| `stratified-group` | `build_patient_group_splits()` in [Preprocess_kfold_feature_selection.py](../../Source_code/main/diabetic_retinopathy/Preprocess_kfold_feature_selection.py) | scikit-learn `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. Produces a different partition. The runner rejects a fallback to plain `GroupKFold`. |

The existing packaged folds are **custom patient-balanced folds**, not SGKF.
Both methods preserve each patient's images in one held-out fold while retaining
the label of each individual image. See [PROTOCOL.md](PROTOCOL.md) for wording
that distinguishes the two designs.

## Verified results, 2026-11-01

| Check | Result |
|---|---|
| Recreate existing custom subset | All 1,208 image-to-fold assignments match; 729 patients. |
| Reload the existing external manifests | All 1,208 assignments match, checked separately against the source loader. |
| Repeat with seed 42 | Identical assignments for each algorithm on the subset. |
| Direct SGKF on the subset | Patient overlap 0; each image held out once; partition differs from the custom reference. |
| SGKF on the complete local label table | 16,266 image records / 8,524 patients; patient overlap 0; each image held out once. |
| Automated tests | 4 tests pass: grouping and per-image labels, dirty-record filtering, reference mismatch, output preservation. |

Aggregate evidence is in [validation_summary.json](validation_summary.json).
The complete label-table check verifies **ID partitioning**. It does not check
that all image files are available or establish full-BRSET model performance.
No FKG-E training, hyperparameter selection, SMOTE, feature fitting or PR-AUC
experiment was run for this release. The earlier FKG-E validation issues are
outside the verified scope of this folder.

## Environment

The checks used Python **3.13.7** and the versions pinned in
[requirements-lock.txt](requirements-lock.txt). Hardware and OS are recorded in
[environment.json](environment.json).

Optional isolated setup on Windows, from the repository root:

```powershell
python -m venv reproducibility/brset_patient_folds/.venv
reproducibility/brset_patient_folds/.venv/Scripts/python.exe -m pip install -r reproducibility/brset_patient_folds/requirements-lock.txt
```

Use that interpreter instead of `python` in the commands below when using the
venv. Linux uses `.venv/bin/python`. This folder depends on the original source
files elsewhere in the cloned repository; it is not a standalone download.

## Input and filtering

Provide a CSV with `image_id`, `patient_id`, and a binary per-image target
(`0` = no DR, `1` = DR). The default target column is `diabetic_retinopathy`;
use `--label-column retinopathy` for the earlier root-train manifest.

The wrapper strips ID whitespace and then removes records with missing image
or patient IDs, all copies of duplicate image IDs, and labels outside `{0, 1}`.
Counts are written to `split_report.json`. Remaining input row order is preserved:
the custom algorithm's seeded tie-breaking depends on patient encounter order.
The verified real inputs required **zero removals**. This explicit wrapper
filtering does not change the filtering behavior of the legacy CLI itself.

## Exact commands

These commands generate private ID files under an ignored directory. Supply
your own authorized BRSET CSV when the example source data is unavailable.

Reproduce the existing custom subset and compare every held-out image set:

```powershell
python -B -X utf8 reproducibility/brset_patient_folds/run.py --input Source_code/data/exports/frb_patient_id_20260923/root_split/train.csv --label-column retinopathy --method custom --folds 5 --seed 42 --reference-folds Source_code/data/exports/frb_patient_id_20260923/dataset_manifests/KFold_feature_selection_rerun_20260921/fusion --output reproducibility/brset_patient_folds/private/subset_custom
```

Generate actual SGKF on the complete local BRSET label table:

```powershell
python -B -X utf8 reproducibility/brset_patient_folds/run.py --input ROOT_DATA/labels_brset.csv --method stratified-group --folds 5 --seed 42 --output reproducibility/brset_patient_folds/private/full_labels_sgkf
```

Compare SGKF against the earlier custom subset (a mismatch is expected):

```powershell
python -B -X utf8 reproducibility/brset_patient_folds/run.py --input Source_code/data/exports/frb_patient_id_20260923/root_split/train.csv --label-column retinopathy --method stratified-group --folds 5 --seed 42 --reference-folds Source_code/data/exports/frb_patient_id_20260923/dataset_manifests/KFold_feature_selection_rerun_20260921/fusion --output reproducibility/brset_patient_folds/private/subset_sgkf
```

Run the tests, which use generated fictional IDs:

```powershell
python -B -X utf8 -m unittest discover -s reproducibility/brset_patient_folds/tests -v
```

An output directory must be new or empty. Use a new directory for another run;
the runner refuses to overwrite existing output. A reference mismatch produces
a report and Python exit code 2; invalid input or a split invariant failure exits
with code 1. A mismatch does not mean SGKF leaks patients: the custom and SGKF
algorithms simply produce different partitions.

Each output contains:

- `fold_assignments.csv`: image ID, patient ID, original binary target, fold.
- `fold_01` through `fold_05`: original-image `train_ids.csv` and `test_ids.csv`.
- `split_report.json`: input/source hashes, seed, environment, filtering counts,
  coverage, patient overlap, fold counts and optional reference comparison.

These are ID manifests, not feature matrices. When changing the split, regenerate
all learned preprocessing, SMOTE, fuzzy rules and graph construction from the
training portion of each new fold. Existing feature/rule CSVs belong to their
original split and cannot be relabeled as a new SGKF experiment.

## Publication contents

This release contains code, fictional-data tests, commands, environment versions
and aggregate validation evidence. Patient-level manifests, images, feature or
rule tables, and the supplied BRSET zip archives are excluded from this commit.
Keep generated IDs under `private/` or `outputs/`, both ignored here. The release
commit identifies the public code; input/source SHA256 values identify the local
files used for verification. Raw file hashes can change with Git line-ending
settings, so retain the exact source bytes when matching byte-level hashes.
