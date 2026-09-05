from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_SOURCE_LABEL_COLUMN = "diabetic_retinopathy"
DEFAULT_OUTPUT_LABEL_COLUMN = "retinopathy"
DEFAULT_OUTPUT_DIR_NAME = "train_test_selection"
DEFAULT_TRAIN_FOLDS = 5
MANIFEST_BASE_COLUMNS = ["image_id", "patient_id", "image_path"]


@dataclass(frozen=True)
class PatientGroup:
    patient_id: str
    row_indices: List[int]
    label_counts: Dict[int, int]
    image_count: int
    tie_breaker: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a patient-level train/test split for ROOT_DATA fundus images. "
            "Only image_id, patient_id, and the configured label column are read "
            "from labels_brset.csv; output manifests use retinopathy as the label."
        )
    )
    parser.add_argument("--root-data", type=Path, default=Path("ROOT_DATA"))
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--train-folds",
        type=int,
        default=DEFAULT_TRAIN_FOLDS,
        help="Number of patient-level KFold splits to create inside the train set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-label-column", default=DEFAULT_SOURCE_LABEL_COLUMN)
    parser.add_argument("--output-label-column", default=DEFAULT_OUTPUT_LABEL_COLUMN)
    parser.add_argument(
        "--materialize",
        choices=["hardlink", "copy", "none"],
        default="hardlink",
        help=(
            "How to create split image folders. hardlink saves disk space and falls "
            "back to copy if hardlinks are unavailable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination files with the same names.",
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Skip images that are not present in labels_brset.csv instead of failing.",
    )
    return parser.parse_args()


def resolve_path(path: Path | None, default_path: Path) -> Path:
    if path is None:
        return default_path
    return path if path.is_absolute() else Path.cwd() / path


def list_image_files(image_dir: Path) -> List[Path]:
    image_files = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_files = sorted(
        image_files,
        key=lambda path: path.relative_to(image_dir).as_posix().lower(),
    )
    if not image_files:
        raise FileNotFoundError(f"No supported image files found in {image_dir}")
    return image_files


def read_required_labels(
    labels_csv: Path,
    requested_source_label_column: str,
    output_label_column: str,
) -> tuple[pd.DataFrame, str]:
    labels = pd.read_csv(
        labels_csv,
        dtype={"image_id": "string", "patient_id": "string"},
    )
    for required_column in ("image_id", "patient_id"):
        if required_column not in labels.columns:
            raise ValueError(f"{labels_csv} is missing required column {required_column!r}")

    label_candidates = [requested_source_label_column]
    if output_label_column not in label_candidates:
        label_candidates.append(output_label_column)
    if DEFAULT_SOURCE_LABEL_COLUMN not in label_candidates:
        label_candidates.append(DEFAULT_SOURCE_LABEL_COLUMN)

    source_label_column = next(
        (column for column in label_candidates if column in labels.columns),
        None,
    )
    if source_label_column is None:
        raise ValueError(
            f"{labels_csv} does not contain any label column from {label_candidates}"
        )

    label_frame = labels[["image_id", "patient_id", source_label_column]].copy()
    label_frame["image_id"] = label_frame["image_id"].astype("string").str.strip()
    label_frame["patient_id"] = label_frame["patient_id"].astype("string").str.strip()
    label_frame[output_label_column] = pd.to_numeric(
        label_frame[source_label_column],
        errors="coerce",
    )

    missing_label = label_frame[label_frame[output_label_column].isna()]
    if not missing_label.empty:
        examples = ", ".join(missing_label["image_id"].head(5).astype(str))
        raise ValueError(f"Missing or non-numeric labels for image_id values: {examples}")

    non_integer = label_frame[
        label_frame[output_label_column].map(lambda value: float(value).is_integer())
        == False
    ]
    if not non_integer.empty:
        examples = ", ".join(non_integer["image_id"].head(5).astype(str))
        raise ValueError(f"Non-integer labels for image_id values: {examples}")

    duplicate_image_ids = label_frame[label_frame.duplicated("image_id", keep=False)]
    if not duplicate_image_ids.empty:
        examples = ", ".join(duplicate_image_ids["image_id"].head(5).astype(str))
        raise ValueError(f"Duplicate image_id values in labels CSV: {examples}")

    label_frame[output_label_column] = label_frame[output_label_column].astype(int)
    return label_frame[["image_id", "patient_id", output_label_column]], source_label_column


def build_labeled_image_frame(
    image_files: Sequence[Path],
    labels: pd.DataFrame,
    output_label_column: str,
    allow_unlabeled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_frame = pd.DataFrame(
        {
            "image_id": [path.stem for path in image_files],
            "source_image_path": [str(path.resolve()) for path in image_files],
        }
    )
    duplicate_image_ids = image_frame[image_frame.duplicated("image_id", keep=False)]
    if not duplicate_image_ids.empty:
        examples = ", ".join(duplicate_image_ids["image_id"].head(5).astype(str))
        raise ValueError(f"Duplicate image_id values from image filenames: {examples}")

    merged = image_frame.merge(labels, on="image_id", how="left")
    unmatched = merged[
        merged["patient_id"].isna() | merged[output_label_column].isna()
    ].copy()
    if not unmatched.empty and not allow_unlabeled:
        examples = ", ".join(unmatched["image_id"].head(10).astype(str))
        raise ValueError(
            "Some images could not be labeled from labels_brset.csv. "
            f"Examples: {examples}. Re-run with --allow-unlabeled to skip them."
        )

    labeled = merged.dropna(subset=["patient_id", output_label_column]).copy()
    labeled["patient_id"] = labeled["patient_id"].astype(str)
    labeled[output_label_column] = labeled[output_label_column].astype(int)
    labeled = labeled.reset_index(drop=True)
    return labeled, unmatched


def add_counts(left: Dict[int, int], right: Dict[int, int]) -> Dict[int, int]:
    result = dict(left)
    for label, count in right.items():
        result[label] = result.get(label, 0) + count
    return result


def split_error(
    patient_count: int,
    image_count: int,
    label_counts: Dict[int, int],
    target_patient_count: float,
    target_image_count: float,
    target_label_counts: Dict[int, float],
    label_values: Iterable[int],
) -> float:
    patient_scale = max(1.0, target_patient_count)
    image_scale = max(1.0, target_image_count)
    error = ((patient_count - target_patient_count) / patient_scale) ** 2
    error += ((image_count - target_image_count) / image_scale) ** 2
    for label in label_values:
        target_label_count = target_label_counts[label]
        label_scale = max(1.0, target_label_count)
        actual_label_count = label_counts.get(label, 0)
        error += ((actual_label_count - target_label_count) / label_scale) ** 2
    return error


def make_patient_groups(
    frame: pd.DataFrame,
    output_label_column: str,
    seed: int,
) -> List[PatientGroup]:
    rng = random.Random(seed)
    groups: List[PatientGroup] = []
    for patient_id, patient_rows in frame.groupby("patient_id", sort=False):
        label_counts = {
            int(label): int(count)
            for label, count in patient_rows[output_label_column]
            .value_counts()
            .sort_index()
            .items()
        }
        groups.append(
            PatientGroup(
                patient_id=str(patient_id),
                row_indices=list(patient_rows.index),
                label_counts=label_counts,
                image_count=int(len(patient_rows)),
                tie_breaker=rng.random(),
            )
        )
    return groups


def split_by_patient(
    frame: pd.DataFrame,
    output_label_column: str,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1")

    patient_groups = make_patient_groups(frame, output_label_column, seed)
    if len(patient_groups) < 2:
        raise ValueError("Patient-level split requires at least two unique patient_id values")

    label_values = sorted(int(value) for value in frame[output_label_column].unique())
    total_label_counts = {
        int(label): int(count)
        for label, count in frame[output_label_column].value_counts().sort_index().items()
    }
    target_patient_count = len(patient_groups) * test_ratio
    target_image_count = len(frame) * test_ratio
    target_label_counts = {
        label: total_label_counts.get(label, 0) * test_ratio for label in label_values
    }

    remaining = list(patient_groups)
    test_patient_ids: set[str] = set()
    test_label_counts = {label: 0 for label in label_values}
    test_image_count = 0
    current_error = split_error(
        0,
        0,
        test_label_counts,
        target_patient_count,
        target_image_count,
        target_label_counts,
        label_values,
    )

    while remaining:
        best_group = None
        best_error = None
        best_counts = None
        for group in remaining:
            next_counts = add_counts(test_label_counts, group.label_counts)
            next_error = split_error(
                len(test_patient_ids) + 1,
                test_image_count + group.image_count,
                next_counts,
                target_patient_count,
                target_image_count,
                target_label_counts,
                label_values,
            )
            if best_error is None or (next_error, group.tie_breaker) < (
                best_error,
                best_group.tie_breaker if best_group else 0.0,
            ):
                best_group = group
                best_error = next_error
                best_counts = next_counts

        if best_group is None or best_counts is None or best_error is None:
            break
        if test_patient_ids and best_error >= current_error:
            break

        test_patient_ids.add(best_group.patient_id)
        test_label_counts = best_counts
        test_image_count += best_group.image_count
        current_error = best_error
        remaining.remove(best_group)

    if not test_patient_ids or len(test_patient_ids) == len(patient_groups):
        raise RuntimeError("Patient-level split produced an empty train or test set")

    output = frame.copy()
    output["split"] = output["patient_id"].map(
        lambda patient_id: "test" if str(patient_id) in test_patient_ids else "train"
    )

    train_patient_ids = set(output.loc[output["split"] == "train", "patient_id"].astype(str))
    test_patient_ids = set(output.loc[output["split"] == "test", "patient_id"].astype(str))
    overlap = train_patient_ids & test_patient_ids
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise RuntimeError(f"Patient id leakage across train/test split: {examples}")
    return output


def patient_group_priority(
    group: PatientGroup,
    total_label_counts: Dict[int, int],
) -> tuple[float, int, float]:
    rarity_score = sum(
        count / max(1, total_label_counts.get(label, 0))
        for label, count in group.label_counts.items()
    )
    return (-rarity_score, -group.image_count, group.tie_breaker)


def assign_train_folds(
    train_frame: pd.DataFrame,
    output_label_column: str,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    if n_folds <= 1:
        raise ValueError("--train-folds must be greater than 1, or set it to 0 to skip")

    fold_frame = train_frame.copy().reset_index(drop=True)
    patient_groups = make_patient_groups(fold_frame, output_label_column, seed + 10_000)
    if len(patient_groups) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds} train folds from only {len(patient_groups)} patients"
        )

    label_values = sorted(int(value) for value in fold_frame[output_label_column].unique())
    total_label_counts = {
        int(label): int(count)
        for label, count in fold_frame[output_label_column].value_counts().sort_index().items()
    }
    target_patient_count = len(patient_groups) / n_folds
    target_image_count = len(fold_frame) / n_folds
    target_label_counts = {
        label: total_label_counts.get(label, 0) / n_folds for label in label_values
    }

    fold_stats = [
        {
            "patient_count": 0,
            "image_count": 0,
            "label_counts": {label: 0 for label in label_values},
        }
        for _ in range(n_folds)
    ]
    assignments: Dict[str, int] = {}

    def primary_label(group: PatientGroup) -> int:
        return min(
            group.label_counts,
            key=lambda label: (total_label_counts.get(label, 0), -group.label_counts[label]),
        )

    groups_by_label: Dict[int, List[PatientGroup]] = {label: [] for label in label_values}
    for group in patient_groups:
        groups_by_label[primary_label(group)].append(group)

    ordered_groups: List[PatientGroup] = []
    for label in sorted(label_values, key=lambda value: total_label_counts.get(value, 0)):
        ordered_groups.extend(
            sorted(
                groups_by_label.get(label, []),
                key=lambda group: patient_group_priority(group, total_label_counts),
            )
        )

    def error_for_stats(stats: Dict[str, object]) -> float:
        return split_error(
            int(stats["patient_count"]),
            int(stats["image_count"]),
            stats["label_counts"],
            target_patient_count,
            target_image_count,
            target_label_counts,
            label_values,
        )

    def error_after_assignment(group: PatientGroup, fold_index: int) -> float:
        total_error = 0.0
        for index, stats in enumerate(fold_stats):
            if index == fold_index:
                total_error += split_error(
                    int(stats["patient_count"]) + 1,
                    int(stats["image_count"]) + group.image_count,
                    add_counts(stats["label_counts"], group.label_counts),
                    target_patient_count,
                    target_image_count,
                    target_label_counts,
                    label_values,
                )
            else:
                total_error += error_for_stats(stats)
        return total_error

    def assign_group(group: PatientGroup, fold_index: int) -> None:
        assignments[group.patient_id] = fold_index + 1
        fold_stats[fold_index]["patient_count"] += 1
        fold_stats[fold_index]["image_count"] += group.image_count
        fold_stats[fold_index]["label_counts"] = add_counts(
            fold_stats[fold_index]["label_counts"],
            group.label_counts,
        )

    for group in ordered_groups:
        best_fold_index = None
        best_key = None
        for fold_index, stats in enumerate(fold_stats):
            key = (
                error_after_assignment(group, fold_index),
                stats["image_count"],
                stats["patient_count"],
                fold_index,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_fold_index = fold_index
        if best_fold_index is None:
            raise RuntimeError("Could not assign patient group to a train fold")
        assign_group(group, best_fold_index)

    fold_frame["fold"] = fold_frame["patient_id"].map(
        lambda patient_id: assignments[str(patient_id)]
    )
    if fold_frame["fold"].isna().any():
        raise RuntimeError("Some train rows were not assigned to a fold")

    for fold_number in range(1, n_folds + 1):
        fold_patients = set(
            fold_frame.loc[fold_frame["fold"] == fold_number, "patient_id"].astype(str)
        )
        if not fold_patients:
            raise RuntimeError(f"Fold {fold_number} has no validation patients")
        other_patients = set(
            fold_frame.loc[fold_frame["fold"] != fold_number, "patient_id"].astype(str)
        )
        overlap = fold_patients & other_patients
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise RuntimeError(f"Patient id leakage in fold {fold_number}: {examples}")

    fold_frame["fold"] = fold_frame["fold"].astype(int)
    return fold_frame


def materialize_image(
    source_path: Path,
    destination_path: Path,
    mode: str,
    overwrite: bool,
) -> str:
    if mode == "none":
        return "not_materialized"

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if overwrite:
            destination_path.unlink()
        else:
            return "existing"

    if mode == "copy":
        shutil.copy2(source_path, destination_path)
        return "copied"

    try:
        os.link(source_path, destination_path)
        return "hardlinked"
    except OSError:
        shutil.copy2(source_path, destination_path)
        return "copied_after_hardlink_failed"


def empty_operations() -> Dict[str, int]:
    return {
        "hardlinked": 0,
        "copied": 0,
        "copied_after_hardlink_failed": 0,
        "existing": 0,
        "not_materialized": 0,
    }


def add_operations(total: Dict[str, int], current: Dict[str, int]) -> Dict[str, int]:
    for operation, count in current.items():
        total[operation] = total.get(operation, 0) + count
    return total


def materialize_manifest_records(
    frame: pd.DataFrame,
    output_dir: Path,
    output_label_column: str,
    materialize: str,
    overwrite: bool,
    split_column: str,
    split_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    operations = empty_operations()
    records = []
    for _, row in frame.iterrows():
        source_path = Path(row["source_image_path"])
        label = int(row[output_label_column])
        split_name = str(row[split_column])
        file_name = f"{row['image_id']}{source_path.suffix}"
        destination_path = (
            output_dir / split_name / str(label) / file_name
        )
        if materialize != "none" and split_names:
            for peer_split_name in split_names:
                peer_path = output_dir / peer_split_name / str(label) / file_name
                if peer_path != destination_path and peer_path.exists():
                    peer_path.unlink()
        operation = materialize_image(source_path, destination_path, materialize, overwrite)
        operations[operation] = operations.get(operation, 0) + 1
        image_path = source_path if materialize == "none" else destination_path
        records.append(
            {
                "image_id": row["image_id"],
                "patient_id": str(row["patient_id"]),
                "split": split_name,
                "image_path": str(image_path.resolve()),
                output_label_column: label,
            }
        )
    return pd.DataFrame(records), operations


def write_split_outputs(
    split_frame: pd.DataFrame,
    unmatched: pd.DataFrame,
    output_dir: Path,
    output_label_column: str,
    materialize: str,
    overwrite: bool,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, operations = materialize_manifest_records(
        split_frame,
        output_dir,
        output_label_column,
        materialize,
        overwrite,
        "split",
        ("train", "test"),
    )
    base_columns = MANIFEST_BASE_COLUMNS + [output_label_column]
    train_manifest = manifest[manifest["split"] == "train"][base_columns].sort_values("image_id")
    test_manifest = manifest[manifest["split"] == "test"][base_columns].sort_values("image_id")
    all_manifest = manifest[
        ["image_id", "patient_id", "split", "image_path", output_label_column]
    ].sort_values(["split", "image_id"])

    train_manifest.to_csv(output_dir / "train.csv", index=False)
    test_manifest.to_csv(output_dir / "test.csv", index=False)
    all_manifest.to_csv(output_dir / "all_images.csv", index=False)

    unmatched_columns = ["image_id", "source_image_path"]
    unmatched.reindex(columns=unmatched_columns).rename(
        columns={"source_image_path": "image_path"}
    ).to_csv(output_dir / "unmatched_images.csv", index=False)

    return operations


def write_train_kfold_outputs(
    train_frame: pd.DataFrame,
    output_dir: Path,
    output_label_column: str,
    n_folds: int,
    seed: int,
    materialize: str,
    overwrite: bool,
) -> Dict[str, object]:
    if n_folds == 0:
        return {"enabled": False}

    fold_root = output_dir / "train_kfold"
    fold_root.mkdir(parents=True, exist_ok=True)
    fold_frame = assign_train_folds(train_frame, output_label_column, n_folds, seed)

    assignment_columns = ["image_id", "patient_id", "fold", "source_image_path", output_label_column]
    fold_frame[assignment_columns].rename(columns={"source_image_path": "image_path"}).sort_values(
        ["fold", "image_id"]
    ).to_csv(fold_root / "fold_assignments.csv", index=False)

    base_columns = MANIFEST_BASE_COLUMNS + [output_label_column]
    total_operations = empty_operations()
    fold_summaries: List[Dict[str, object]] = []
    for fold_number in range(1, n_folds + 1):
        fold_dir = fold_root / f"fold_{fold_number}"
        fold_split_frame = fold_frame.copy()
        fold_split_frame["fold_split"] = fold_split_frame["fold"].map(
            lambda assigned_fold: "val" if assigned_fold == fold_number else "train"
        )
        manifest, operations = materialize_manifest_records(
            fold_split_frame,
            fold_dir,
            output_label_column,
            materialize,
            overwrite,
            "fold_split",
            ("train", "val"),
        )
        add_operations(total_operations, operations)

        train_manifest = manifest[manifest["split"] == "train"][base_columns].sort_values(
            "image_id"
        )
        val_manifest = manifest[manifest["split"] == "val"][base_columns].sort_values(
            "image_id"
        )
        all_manifest = manifest[
            ["image_id", "patient_id", "split", "image_path", output_label_column]
        ].sort_values(["split", "image_id"])
        train_manifest.to_csv(fold_dir / "train.csv", index=False)
        val_manifest.to_csv(fold_dir / "val.csv", index=False)
        all_manifest.to_csv(fold_dir / "all_images.csv", index=False)

        train_patients = set(train_manifest["patient_id"].astype(str))
        val_patients = set(val_manifest["patient_id"].astype(str))
        fold_summaries.append(
            {
                "fold": fold_number,
                "output_dir": str(fold_dir.resolve()),
                "patient_overlap_count": int(len(train_patients & val_patients)),
                "train": count_summary(train_manifest, output_label_column),
                "val": count_summary(val_manifest, output_label_column),
            }
        )

    summary = {
        "enabled": True,
        "output_dir": str(fold_root.resolve()),
        "fold_count": n_folds,
        "source_split": "train",
        "manifest_columns": base_columns,
        "folds": fold_summaries,
        "file_operations": total_operations,
    }
    with (fold_root / "kfold_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")
    return summary


def count_summary(frame: pd.DataFrame, output_label_column: str) -> Dict[str, object]:
    return {
        "images": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "retinopathy_counts": {
            str(label): int(count)
            for label, count in frame[output_label_column].value_counts().sort_index().items()
        },
    }


def main() -> int:
    args = parse_args()
    root_data = resolve_path(args.root_data, Path("ROOT_DATA")).resolve()
    labels_csv = resolve_path(args.labels_csv, root_data / "labels_brset.csv").resolve()
    image_dir = resolve_path(args.image_dir, root_data / "fundus_photos").resolve()
    output_dir = resolve_path(
        args.output_dir,
        root_data / DEFAULT_OUTPUT_DIR_NAME,
    ).resolve()

    labels, source_label_column = read_required_labels(
        labels_csv,
        args.source_label_column,
        args.output_label_column,
    )
    image_files = list_image_files(image_dir)
    labeled_images, unmatched = build_labeled_image_frame(
        image_files,
        labels,
        args.output_label_column,
        args.allow_unlabeled,
    )
    split_frame = split_by_patient(
        labeled_images,
        args.output_label_column,
        args.test_ratio,
        args.seed,
    )
    operations = write_split_outputs(
        split_frame,
        unmatched,
        output_dir,
        args.output_label_column,
        args.materialize,
        args.overwrite,
    )

    train_frame = split_frame[split_frame["split"] == "train"]
    test_frame = split_frame[split_frame["split"] == "test"]
    train_kfold = write_train_kfold_outputs(
        train_frame,
        output_dir,
        args.output_label_column,
        args.train_folds,
        args.seed,
        args.materialize,
        args.overwrite,
    )
    train_patients = set(train_frame["patient_id"].astype(str))
    test_patients = set(test_frame["patient_id"].astype(str))
    summary = {
        "root_data": str(root_data),
        "labels_csv": str(labels_csv),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "source_label_column": source_label_column,
        "output_label_column": args.output_label_column,
        "manifest_columns": MANIFEST_BASE_COLUMNS + [args.output_label_column],
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "materialize": args.materialize,
        "image_count": int(len(image_files)),
        "labeled_image_count": int(len(labeled_images)),
        "unmatched_image_count": int(len(unmatched)),
        "patient_count": int(labeled_images["patient_id"].nunique()),
        "patient_overlap_count": int(len(train_patients & test_patients)),
        "train": count_summary(train_frame, args.output_label_column),
        "test": count_summary(test_frame, args.output_label_column),
        "train_kfold": train_kfold,
        "file_operations": operations,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
