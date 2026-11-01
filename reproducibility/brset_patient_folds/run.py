"""Generate and verify patient fold IDs using the existing repository split code.

No feature fitting, SMOTE, image processing, rule mining, or model training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "Source_code/main/diabetic_retinopathy"
METHODS = ("custom", "stratified-group")
SOURCE_FILES = {
    "custom": "create_root_data_image_train_test_split.py",
    "stratified-group": "Preprocess_kfold_feature_selection.py",
}


def clean_records(frame: pd.DataFrame, label_column: str):
    """Keep real image IDs, patient IDs and binary image labels; preserve row order.

    All remaining copies of a duplicate image ID are removed, including conflicts.
    Counts are sequential: missing IDs, duplicate IDs, then invalid labels.
    """
    required = ["image_id", "patient_id", label_column]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame[required].copy()
    audit = {"input_rows": len(data)}
    for column in ("image_id", "patient_id"):
        data[column] = data[column].astype("string").str.strip()
    invalid_ids = data[["image_id", "patient_id"]].isna().any(axis=1)
    invalid_ids |= data[["image_id", "patient_id"]].eq("").any(axis=1)
    audit["removed_missing_ids"] = int(invalid_ids.sum())
    data = data.loc[~invalid_ids].copy()
    duplicate = data["image_id"].duplicated(keep=False)
    audit["removed_duplicate_image_ids"] = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    label = pd.to_numeric(data[label_column], errors="coerce")
    valid = label.isin([0, 1])
    audit["removed_invalid_binary_labels"] = int((~valid).sum())
    data = data.loc[valid].copy()
    data[label_column] = label.loc[valid].astype(int)
    data = data.reset_index(drop=True)
    audit["retained_images"] = len(data)
    audit["retained_patients"] = int(data["patient_id"].nunique())
    if data.empty:
        raise ValueError("No valid image records remain after filtering.")
    return data, audit


def _source_module(method: str):
    """Import the original source without invoking its guarded CLI entrypoint."""
    name = "_brset_fold_release_" + method.replace("-", "_")
    source = LEGACY / SOURCE_FILES[method]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def create_assignments(data, label_column, method="custom", folds=5, seed=42):
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if folds < 2 or data["patient_id"].nunique() < folds:
        raise ValueError("Need at least two folds and at least one patient per fold.")
    module = _source_module(method)
    if method == "custom":
        assigned = module.assign_train_folds(data, label_column, folds, seed)
        implementation = "assign_train_folds: custom patient-balanced allocation"
    else:
        # The splitter uses X only for row count; no feature values are fitted.
        x = pd.DataFrame({"row_count_only": np.zeros(len(data))})
        actual, _, splits = module.build_patient_group_splits(
            x, data[label_column], data[["image_id", "patient_id"]], folds, seed
        )
        if actual != "StratifiedGroupKFold":
            raise RuntimeError(f"Requested SGKF, but legacy code returned {actual}.")
        assigned = data.copy()
        assigned["fold"] = 0
        appearances = np.zeros(len(data), dtype=int)
        for fold, (train_index, test_index) in enumerate(splits, 1):
            if set(train_index) & set(test_index):
                raise AssertionError("Image overlap in splitter output.")
            if set(train_index) | set(test_index) != set(range(len(data))):
                raise AssertionError("A splitter fold does not cover all retained rows.")
            assigned.loc[test_index, "fold"] = fold
            appearances[test_index] += 1
        if not (appearances == 1).all():
            raise AssertionError("Each image must appear in held-out exactly once.")
        implementation = actual
    return assigned, implementation


def verify_assignments(assigned, label_column, folds):
    if assigned["image_id"].duplicated().any():
        raise AssertionError("Duplicate image ID in assignments.")
    if not assigned["fold"].isin(range(1, folds + 1)).all():
        raise AssertionError("Unassigned image or invalid fold number.")
    if (assigned.groupby("patient_id")["fold"].nunique() != 1).any():
        raise AssertionError("A patient's images cross held-out folds.")
    details = []
    for fold in range(1, folds + 1):
        test = assigned[assigned["fold"] == fold]
        train = assigned[assigned["fold"] != fold]
        overlap = set(train["patient_id"]) & set(test["patient_id"])
        if test.empty or train.empty or overlap:
            raise AssertionError(f"Invalid or leaking fold {fold}.")
        details.append({
            "fold": fold,
            "train_images": len(train),
            "test_images": len(test),
            "train_patients": int(train["patient_id"].nunique()),
            "test_patients": int(test["patient_id"].nunique()),
            "test_positives": int(test[label_column].sum()),
            "patient_overlap_count": len(overlap),
        })
    return {
        "each_image_held_out_once": True,
        "each_patient_in_one_held_out_fold": True,
        "patient_overlap_count": 0,
        "folds": details,
    }


def compare_reference(assigned, reference_root, folds):
    checks = []
    retained_ids = set(assigned["image_id"])
    reference_union = set()
    for fold in range(1, folds + 1):
        candidates = [reference_root / f"fold_{fold:02}", reference_root / f"fold_{fold}"]
        directory = next((path for path in candidates if path.is_dir()), None)
        if directory is None:
            raise ValueError(f"Missing reference fold {fold} under {reference_root}")
        path = next((directory / name for name in ("test_ids.csv", "val.csv", "test.csv")
                     if (directory / name).is_file()), None)
        if path is None:
            raise ValueError(f"Missing reference held-out ID file in {directory}")
        ref = pd.read_csv(path, dtype={"image_id": "string"})["image_id"].str.strip()
        if ref.isna().any() or ref.eq("").any() or ref.duplicated().any():
            raise ValueError(f"Invalid or duplicate image IDs in {path}")
        expected = set(ref)
        if reference_union & expected:
            raise ValueError("A reference image appears in multiple held-out folds.")
        reference_union |= expected
        actual = set(assigned.loc[assigned["fold"] == fold, "image_id"])
        checks.append({
            "fold": fold,
            "missing_test_images": len(expected - actual),
            "unexpected_test_images": len(actual - expected),
            "matches": actual == expected,
        })
    coverage_matches = retained_ids == reference_union
    return {
        "all_test_image_sets_match": coverage_matches and all(row["matches"] for row in checks),
        "reference_covers_retained_images": coverage_matches,
        "folds": checks,
    }


def environment():
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl",
                "python-dateutil", "pytz", "tzdata", "six"]
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def run(input_path, output, label_column, method, folds, seed, reference=None):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    frame = pd.read_csv(input_path, dtype={"image_id": "string", "patient_id": "string"})
    clean, filtering = clean_records(frame, label_column)
    assigned, implementation = create_assignments(clean, label_column, method, folds, seed)
    checks = verify_assignments(assigned, label_column, folds)
    # Prove that assignment has not changed any per-image target or patient ID.
    if not assigned[["image_id", "patient_id", label_column]].equals(clean):
        raise AssertionError("Splitter changed original image IDs, patients, labels or row order.")
    comparison = compare_reference(assigned, reference, folds) if reference is not None else None
    source = LEGACY / SOURCE_FILES[method]
    report = {
        "scope": "Patient ID partitioning only; no image availability check or model evaluation.",
        "method": method,
        "implementation": implementation,
        "fold_count": folds,
        "seed": seed,
        "label_column": label_column,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source": source.relative_to(REPO).as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "filtering": filtering,
        "checks": checks,
        "reference_comparison": comparison,
        "environment": environment(),
    }
    output.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(output / "fold_assignments.csv", index=False)
    for fold in range(1, folds + 1):
        directory = output / f"fold_{fold:02}"
        directory.mkdir()
        for name, mask in [("train_ids.csv", assigned["fold"] != fold),
                           ("test_ids.csv", assigned["fold"] == fold)]:
            assigned.loc[mask, ["patient_id", "image_id", label_column]].to_csv(directory / name, index=False)
    (output / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Private CSV with image_id, patient_id and binary label.")
    parser.add_argument("--output", type=Path, required=True, help="New/empty directory for generated private IDs.")
    parser.add_argument("--label-column", default="diabetic_retinopathy")
    parser.add_argument("--method", choices=METHODS, default="custom")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-folds", type=Path, help="Compare against existing fold_XX/test_ids.csv or fold_X/val.csv.")
    args = parser.parse_args()
    try:
        report = run(args.input, args.output, args.label_column, args.method, args.folds, args.seed, args.reference_folds)
    except (ValueError, RuntimeError, AssertionError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    comparison = report["reference_comparison"]
    if comparison is not None and not comparison["all_test_image_sets_match"]:
        parser.exit(2, "Generated partition differs from the reference; see split_report.json.\n")


if __name__ == "__main__":
    main()
