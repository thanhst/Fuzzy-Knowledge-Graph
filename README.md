# FKG-MM: complete reviewer reproduction

This compact branch reruns the complete reported FKG-MM experiment from the
continuous multimodal feature table, rather than starting from saved fuzzy
rules. It needs no compiled extension, Visual Studio, or CUDA.

## Pipeline

For each of the five fixed patient-level folds, `run.py` performs:

1. join feature rows to image and patient IDs;
2. verify zero train/validation patient overlap;
3. fit train-only standardization independently for image and tabular features;
4. select 7 image and 9 tabular features with train-only ANOVA F scores;
5. apply BorderlineSMOTE only to the training fold;
6. generate fuzzy rules with 1-D fuzzy C-means (five clusters per feature);
7. train and evaluate FKG-MM; and
8. write predictions, metrics, intermediate data, selected features, and rules.

The bundled archived rules and native predictions are used only as an
independent equality check. The pipeline does not read them to make a
prediction.

## Run

```bash
git clone --branch MM --single-branch https://github.com/thanhst/Fuzzy-Knowledge-Graph.git
cd Fuzzy-Knowledge-Graph
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --check-reference
```

On Windows, `run.bat` is the one-command entry point after dependencies are
installed. A verified run ends with:

```text
accuracy       91.7 +/-  0.3 %
f1             16.6 +/- 12.9 %
auc_roc        80.7 +/-  6.6 %
specificity    98.7 +/-  0.9 %
sensitivity    11.4 +/- 10.2 %
reference rules match: True
reference predictions match: True
```

Outputs are written below `outputs/latest/`. Use `--folds 1` for a short
single-fold check.

## Included data

- `data/fusion_features.csv`: all continuous image and tabular features;
- `data/row_ids.csv` and `data/labels_brset.csv`: row, image, patient and label provenance;
- `data/splits/`: outer train/test and five patient-grouped folds;
- `reference/fold_01` ... `fold_05`: archived FIS rules for equality checks;
- `results/reference_predictions/`: archived native FKG-MM predictions.

The five-fold experiment uses the 1,208-image outer-train set (729 patients).
Feature extraction from raw fundus images is outside the timed/reported FKG-MM
experiment and is therefore not part of this compact branch.
