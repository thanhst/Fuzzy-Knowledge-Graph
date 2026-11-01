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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

REPO = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent
FOLD_RUNNER = BUNDLE / "portable_folds.py"
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


def prepare_cached(path):
    data = read_csv(path)
    expected = [*IDS, *TABLE_COLUMNS, *IMAGE_COLUMNS]
    if set(data.columns) != set(expected):
        raise ValueError("Cached fusion CSV must contain exactly the IDs, target and 23 feature columns")
    tools = load_fold_tools()
    clean, filtering = tools.clean_records(data, TARGET)
    if len(clean) != len(data):
        raise ValueError("Cached fusion CSV contains invalid IDs/labels")
    data = data[expected].copy()
    data[IDS] = clean
    for column in [*TABLE_COLUMNS, *IMAGE_COLUMNS]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[IMAGE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Cached image features contain missing/nonfinite values")
    if np.isinf(data[TABLE_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError("Clinical features contain infinite values")
    return data, filtering, tools


def source_info(path):
    path = path.resolve()
    return {"path": path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path),
            "sha256": digest(path)}


def run(image_path, label_path, output, folds=5, seed=42, source=None):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new/empty output directory; existing release data will not be overwritten")
    data, filtering, tools = prepare_cached(source) if source else prepare(image_path, label_path)
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
        "sources": {"cached_fusion": source_info(source)} if source else {
            "images": source_info(image_path), "labels": source_info(label_path)},
        "code_sha256": {"builder": digest(Path(__file__)), "fold_runner": digest(FOLD_RUNNER)},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "files_sha256": {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*.csv"))},
    }
    (output / "release_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(json.dumps({"images": len(data), "patients": int(data["patient_id"].nunique()),
                      "method": implementation, "folds": checks["folds"], "output": str(output)}, ensure_ascii=False, indent=2))
    return report


def verify_release(directory):
    """Check published bytes and reproduce its splits with the installed sklearn."""
    manifest = json.loads((directory / "release_manifest.json").read_text(encoding="utf-8"))
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob("*.csv")}
    if actual_files != set(manifest["files_sha256"]):
        raise AssertionError("Release CSV file list differs from manifest")
    for name, expected_hash in manifest["files_sha256"].items():
        if digest(directory / name) != expected_hash:
            raise AssertionError(f"CSV hash mismatch: {name}")
    data, audit, tools = prepare_cached(directory / "fusion_features.csv")
    for key in ("retained_images", "retained_patients"):
        if audit[key] != manifest["filtering"][key]:
            raise AssertionError(f"Population mismatch: {key}")
    assigned = read_csv(directory / "fold_assignments.csv")
    checks = tools.verify_assignments(assigned, TARGET, manifest["fold_count"])
    rebuilt, _ = tools.create_assignments(data[IDS], TARGET, folds=manifest["fold_count"], seed=manifest["seed"])
    pd.testing.assert_frame_equal(rebuilt, assigned)
    for fold in range(1, manifest["fold_count"] + 1):
        for name, mask in (("train", assigned.fold != fold), ("test", assigned.fold == fold)):
            stored = read_csv(directory / f"fold_{fold:02}" / f"{name}_ids.csv")
            pd.testing.assert_frame_equal(stored, assigned.loc[mask, IDS].reset_index(drop=True))
    return {"images": len(data), "patients": audit["retained_patients"],
            "csv_files_verified": len(actual_files), **checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Fusion CSV; defaults to the bundled full dataset")
    parser.add_argument("--image-source", type=Path, help="Optional original raw image-feature CSV; requires --labels")
    parser.add_argument("--labels", type=Path, help="Original label CSV; requires --image-source")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="Verify all bundled CSV hashes and regenerate splits in memory")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.verify_only:
        if args.source or args.image_source or args.labels or args.output:
            parser.error("--verify-only validates the bundled release and cannot be combined with source/output arguments")
        print(json.dumps(verify_release(BUNDLE / "data"), ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error("--output is required when rebuilding")
    if bool(args.image_source) != bool(args.labels) or (args.source and args.image_source):
        parser.error("Use either --source or the pair --image-source and --labels")
    source = (args.source or BUNDLE / "data/fusion_features.csv").resolve() if not args.image_source else None
    run(args.image_source, args.labels, args.output.resolve(), args.folds, args.seed, source=source)


if __name__ == "__main__":
    main()
