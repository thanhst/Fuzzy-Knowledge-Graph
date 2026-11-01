"""Export full-BRSET cached image features and patient-disjoint fold CSVs.

Joins use image_id, never an assumed common row order. No imputation, scaling,
selection, SMOTE, fuzzification, rule learning or model fitting is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

REPO = Path(__file__).resolve().parents[2]
IMAGE_SOURCE = REPO / "Source_code/data/Dataset_diabetic/Fusion_feature/images_ft.csv"
LABEL_SOURCE = REPO / "ROOT_DATA/labels_brset.csv"
FOLD_RUNNER = REPO / "reproducibility/brset_patient_folds/run.py"
EXTRACTOR = REPO / "Source_code/module/Processing_Data/Diabetic_fusion_processing.py"
TARGET = "diabetic_retinopathy"
IDS = ["image_id", "patient_id", TARGET]
IMAGE_COLUMNS = ["Contrast Feature", "Dissimilarity Feature", "Homogeneity Feature", "Energy Feature",
                 "Correlation Feature", "ASM Feature", "Mean Feature", "Variance Feature",
                 "Standard Deviation Feature", "RMS Feature"]
TABLE_COLUMNS = ["patient_age", "patient_sex", "diabetes_time_y", "insuline", "diabetes", "exam_eye",
                 "optic_disc", "vessels", "macula", "focus", "Illuminaton", "image_field", "quality"]
CATEGORY_MAPS = {"insuline": {"yes": 2, "no": 1}, "diabetes": {"yes": 2, "no": 1},
                 "quality": {"adequate": 2, "inadequate": 1}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def read_csv(path):
    frame = pd.read_csv(path, dtype={"image_id": "string", "patient_id": "string"},
                        float_precision="round_trip")
    frame["image_id"] = frame["image_id"].str.strip()
    if frame["image_id"].isna().any() or frame["image_id"].eq("").any() or frame["image_id"].duplicated().any():
        raise ValueError(f"Missing/duplicate image_id in {path}")
    return frame


def load_fold_tools():
    spec = importlib.util.spec_from_file_location("_full_brset_fold_tools", FOLD_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare(image_path, label_path):
    images, labels = read_csv(image_path), read_csv(label_path)
    tools = load_fold_tools()
    clean_ids, filtering = tools.clean_records(labels, TARGET)
    if len(clean_ids) != len(labels):
        raise ValueError("Label filtering removed records; review the population before publishing a full-data release.")
    for column in IMAGE_COLUMNS:
        images[column] = pd.to_numeric(images[column], errors="raise")
    if not np.isfinite(images[IMAGE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Cached image features contain missing/nonfinite values")
    if set(images["image_id"]) != set(clean_ids["image_id"]):
        raise ValueError("Cached image features do not cover the complete retained label population")
    # The source clinical table is read again, retaining missing values from
    # labels rather than using a historical globally imputed clinical CSV.
    clinical = labels[["image_id", *TABLE_COLUMNS]].copy()
    for column in TABLE_COLUMNS:
        if column in CATEGORY_MAPS:
            clinical[column] = clinical[column].astype("string").str.strip().str.lower().map(CATEGORY_MAPS[column])
        else:
            clinical[column] = pd.to_numeric(clinical[column], errors="coerce")
        clinical[column] = clinical[column].replace([np.inf, -np.inf], np.nan)
    data = clean_ids.merge(images[["image_id", *IMAGE_COLUMNS]], on="image_id", validate="one_to_one")
    data = data.merge(clinical, on="image_id", validate="one_to_one")
    # Preserve the original label-table order, as in the published full-label
    # SGKF check. Every feature join above is explicitly keyed by image ID.
    if data[IDS].reset_index(drop=True).equals(clean_ids) is False:
        raise AssertionError("Joining features changed original IDs, targets or row order")
    return data, filtering, tools


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def run(image_path, label_path, output, folds=5, seed=42):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new/empty output directory; existing release data will not be overwritten")
    data, filtering, tools = prepare(image_path, label_path)
    assigned, implementation = tools.create_assignments(data[IDS].copy(), TARGET, "stratified-group", folds, seed)
    checks = tools.verify_assignments(assigned, TARGET, folds)
    data = data.merge(assigned[["image_id", "fold"]], on="image_id", validate="one_to_one")
    output.mkdir(parents=True, exist_ok=True)
    # Keep split metadata in fold_assignments.csv so a generic numeric loader
    # cannot accidentally train on the fold number as an input feature.
    metadata = IDS
    write_csv(output / "image_features.csv", data[[*metadata, *IMAGE_COLUMNS]])
    write_csv(output / "table_features.csv", data[[*metadata, *TABLE_COLUMNS]])
    write_csv(output / "fusion_features.csv", data[[*metadata, *TABLE_COLUMNS, *IMAGE_COLUMNS]])
    write_csv(output / "fold_assignments.csv", assigned)
    for fold in range(1, folds + 1):
        directory = output / f"fold_{fold:02}"
        write_csv(directory / "train_ids.csv", assigned.loc[assigned["fold"] != fold, IDS])
        write_csv(directory / "test_ids.csv", assigned.loc[assigned["fold"] == fold, IDS])
    split_source = tools.LEGACY / tools.SOURCE_FILES["stratified-group"]
    report = {
        "scope": "Full cached per-image feature coverage and patient ID partitioning; no model training or raw-image re-extraction.",
        "method": implementation, "fold_count": folds, "seed": seed,
        "feature_counts": {"image": len(IMAGE_COLUMNS), "table": len(TABLE_COLUMNS),
                           "fusion": len(IMAGE_COLUMNS) + len(TABLE_COLUMNS)},
        "label_semantics": {"0": "no diabetic retinopathy", "1": "diabetic retinopathy"},
        "label_counts": {str(key): int(value) for key, value in data[TARGET].value_counts().sort_index().items()},
        "filtering": filtering, "checks": checks,
        "clinical_missing_values_preserved": {key: int(value) for key, value in data[TABLE_COLUMNS].isna().sum().items()},
        "fixed_category_maps": CATEGORY_MAPS,
        "learned_processing_performed": False, "smote_performed": False,
        "image_source": {"path": image_path.relative_to(REPO).as_posix(), "sha256": digest(image_path),
                         "git_blob": git("rev-parse", "HEAD:" + image_path.relative_to(REPO).as_posix()),
                         "last_change_commit": git("log", "-1", "--format=%H", "--", image_path.relative_to(REPO).as_posix())},
        "label_source": {"path": label_path.relative_to(REPO).as_posix(), "sha256": digest(label_path)},
        "code_sha256": {"builder": digest(Path(__file__)), "fold_runner": digest(FOLD_RUNNER),
                        "splitter_source": digest(split_source), "historical_extractor": digest(EXTRACTOR)},
        "base_commit": git("rev-parse", "HEAD"),
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "files_sha256": {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*.csv"))},
    }
    (output / "release_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(json.dumps({"images": len(data), "patients": int(data["patient_id"].nunique()),
                      "method": implementation, "folds": checks["folds"], "output": str(output)}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-source", type=Path, default=IMAGE_SOURCE)
    parser.add_argument("--labels", type=Path, default=LABEL_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.image_source.resolve(), args.labels.resolve(), args.output.resolve(), args.folds, args.seed)


if __name__ == "__main__":
    main()
