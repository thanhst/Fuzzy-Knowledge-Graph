import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on sklearn version.
    StratifiedGroupKFold = None


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
LABEL_COLUMN = "diabetic_retinopathy"
PATIENT_ID_COLUMN = "patient_id"
IMAGE_ID_COLUMN = "image_id"
ID_COLUMNS = [IMAGE_ID_COLUMN, PATIENT_ID_COLUMN, "id"]
DEFAULT_PATIENT_ID_SOURCE = "data/Dataset_diabetic/labels_brset.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


IMAGE_FEATURE_COLUMNS = [
    "Contrast Feature",
    "Dissimilarity Feature",
    "Homogeneity Feature",
    "Energy Feature",
    "Correlation Feature",
    "ASM Feature",
    "Mean Feature",
    "Variance Feature",
    "Standard Deviation Feature",
    "RMS Feature",
]

TABLE_FEATURE_COLUMNS = [
    "patient_age",
    "patient_sex",
    "diabetes_time_y",
    "insuline",
    "diabetes",
    "exam_eye",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "image_field",
    "quality",
]

TABLE_CLUSTER_BY_COLUMN = {
    "patient_age": 3,
    "patient_sex": 2,
    "diabetes_time_y": 4,
    "insuline": 2,
    "diabetes": 2,
    "exam_eye": 2,
    "optic_disc": 2,
    "vessels": 2,
    "macula": 2,
    "focus": 2,
    "Illuminaton": 2,
    "image_field": 2,
    "quality": 2,
}


@dataclass(frozen=True)
class ModalityConfig:
    key: str
    display_name: str
    source_arg: str
    source_default: str


MODALITY_CONFIGS = {
    "image": ModalityConfig(
        key="image",
        display_name="Diabetic Retinopathy Image Feature FT Selection KFold",
        source_arg="image_source",
        source_default="data/Dataset_diabetic/images_ft.csv",
    ),
    "table": ModalityConfig(
        key="table",
        display_name="Diabetic Retinopathy Metadata Feature FT Selection KFold",
        source_arg="table_source",
        source_default="data/Dataset_diabetic/data_process_tabular.csv",
    ),
    "fusion": ModalityConfig(
        key="fusion",
        display_name="Diabetic Retinopathy Fusion Feature FT Selection KFold",
        source_arg="fusion_source",
        source_default="data/Dataset_diabetic/data_process_fusion.csv",
    ),
    "fusion_filter": ModalityConfig(
        key="fusion_filter",
        display_name="Diabetic Retinopathy Fusion Feature Filter KFold",
        source_arg="fusion_source",
        source_default="data/Dataset_diabetic/data_process_fusion.csv",
    ),
    "fusion_hadamard": ModalityConfig(
        key="fusion_hadamard",
        display_name="Diabetic Retinopathy Fusion Feature Hadamard KFold",
        source_arg="fusion_source",
        source_default="data/Dataset_diabetic/data_process_fusion.csv",
    ),
    "fusion_tensor": ModalityConfig(
        key="fusion_tensor",
        display_name="Diabetic Retinopathy Fusion Feature Tensor KFold",
        source_arg="fusion_source",
        source_default="data/Dataset_diabetic/data_process_fusion.csv",
    ),
    "fusion_wrapper": ModalityConfig(
        key="fusion_wrapper",
        display_name="Diabetic Retinopathy Fusion Feature Wrapper KFold",
        source_arg="fusion_source",
        source_default="data/Dataset_diabetic/data_process_fusion.csv",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create KFold Feature Selection data for image, table, and fusion "
            "diabetic retinopathy scenarios. Optionally run FIS for every fold."
        )
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["image", "table", "fusion"],
        choices=["all"] + list(MODALITY_CONFIGS.keys()),
        help="Modalities to prepare. Use 'all' to run image, table, and fusion.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of KFold splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KFold and SMOTE.")
    parser.add_argument("--k-img", type=int, default=7, help="Selected image features.")
    parser.add_argument("--k-tab", type=int, default=9, help="Selected table features for fusion.")
    parser.add_argument("--k-table", type=int, default=13, help="Selected table-only features.")
    parser.add_argument("--filter-corr", type=float, default=0.95, help="Correlation cutoff for fusion_filter.")
    parser.add_argument("--hadamard-dim", type=int, default=5, help="Common dimension for fusion_hadamard.")
    parser.add_argument("--tensor-rank", type=int, default=16, help="SVD rank for fusion_tensor.")
    parser.add_argument("--wrapper-max-img", type=int, default=7, help="Maximum image features for fusion_wrapper.")
    parser.add_argument("--wrapper-max-tab", type=int, default=9, help="Maximum table features for fusion_wrapper.")
    parser.add_argument("--wrapper-min-img", type=int, default=2, help="Minimum image features for fusion_wrapper.")
    parser.add_argument("--wrapper-min-tab", type=int, default=2, help="Minimum table features for fusion_wrapper.")
    parser.add_argument("--wrapper-cv", type=int, default=3, help="Cross-validation folds for fusion_wrapper scoring.")
    parser.add_argument(
        "--wrapper-rf-estimators",
        type=int,
        default=20,
        help="RandomForest estimator count for fusion_wrapper scoring.",
    )
    parser.add_argument(
        "--image-source",
        default=MODALITY_CONFIGS["image"].source_default,
        help="Image feature CSV relative to Source_code or absolute path.",
    )
    parser.add_argument(
        "--table-source",
        default=MODALITY_CONFIGS["table"].source_default,
        help="Table feature CSV relative to Source_code or absolute path.",
    )
    parser.add_argument(
        "--fusion-source",
        default=MODALITY_CONFIGS["fusion"].source_default,
        help="Fusion feature CSV relative to Source_code or absolute path.",
    )
    parser.add_argument(
        "--patient-id-source",
        default=DEFAULT_PATIENT_ID_SOURCE,
        help=(
            "CSV used to recover patient_id when a feature CSV only has image_id "
            "or matches the original diabetic dataset row order."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="data/Dataset_diabetic/KFold_feature_selection",
        help="Output folder relative to Source_code or absolute path.",
    )
    parser.add_argument(
        "--report-root",
        default="data/result/KFold_feature_selection",
        help="Summary report folder relative to Source_code or absolute path.",
    )
    parser.add_argument(
        "--fis-range-source",
        choices=["train", "full"],
        default="train",
        help="Use train fold or full fold data to compute FIS min/max ranges.",
    )
    parser.add_argument(
        "--fis-engine",
        choices=["native", "legacy"],
        default="native",
        help="FIS engine for rule generation. Native uses fisa_module; legacy uses Source_code/module/FIS logic.",
    )
    parser.add_argument(
        "--native-backend",
        choices=["cpu", "gpu", "auto"],
        default="cpu",
        help="Backend used by native fisa_module FIS.",
    )
    parser.add_argument(
        "--skip-fis",
        action="store_true",
        help="Only write KFold train/test CSV files; do not generate FIS rules.",
    )
    parser.add_argument(
        "--skip-fis-test",
        action="store_true",
        help="Generate FIS rules but skip FIS_Test_file evaluation.",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip fold heatmap generation.",
    )
    parser.add_argument(
        "--skip-smote",
        action="store_true",
        help="Do not apply BorderlineSMOTE to each train fold.",
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="Maximum k_neighbors for BorderlineSMOTE.",
    )
    parser.add_argument(
        "--run-fkgs",
        action="store_true",
        help="Run FKGS for every generated FIS fold.",
    )
    parser.add_argument(
        "--run-fkg",
        action="store_true",
        help="Run native FKG on every generated FIS rule fold.",
    )
    parser.add_argument(
        "--only-fkg",
        action="store_true",
        help="Reuse the existing KFold manifest and only run/update FKG results.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Reuse the existing KFold manifest and only rewrite CSV/PNG reports.",
    )
    parser.add_argument(
        "--fkg-backend",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Backend used by native fisa_module FKG.",
    )
    parser.add_argument("--ran", type=int, nargs="+", default=[20], help="FKGS ran values.")
    parser.add_argument("--e", type=float, nargs="+", default=[0.2, 0.3], help="FKGS e values.")
    parser.add_argument(
        "--fkgs-turns",
        type=int,
        default=1,
        help="Number of FKGS random sampling turns per fold/config.",
    )
    parser.add_argument(
        "--reuse-fkgs",
        action="store_true",
        help="Reuse existing per-fold FKGS summary JSON files when present.",
    )
    parser.add_argument(
        "--fkgs-workers",
        type=int,
        default=1,
        help="Maximum parallel FKGS subprocesses per fold.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for a quick smoke run.",
    )
    return parser.parse_args()


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def selected_modalities(modalities):
    if "all" in modalities:
        return ["image", "table", "fusion"]
    return list(dict.fromkeys(modalities))


def normalize_id_series(series):
    return series.map(lambda value: "" if pd.isna(value) else str(value).strip())


def load_patient_id_reference(patient_id_source):
    reference_path = resolve_project_path(patient_id_source)
    if not reference_path.exists():
        raise FileNotFoundError(f"Patient id source CSV not found: {reference_path}")

    reference_df = pd.read_csv(
        reference_path,
        dtype={IMAGE_ID_COLUMN: str, PATIENT_ID_COLUMN: str, "id": str},
    )
    if PATIENT_ID_COLUMN not in reference_df.columns:
        raise ValueError(f"Patient id source must contain '{PATIENT_ID_COLUMN}': {reference_path}")

    reference_df[PATIENT_ID_COLUMN] = normalize_id_series(reference_df[PATIENT_ID_COLUMN])
    if (reference_df[PATIENT_ID_COLUMN] == "").any():
        raise ValueError(f"Patient id source contains empty patient_id values: {reference_path}")
    if IMAGE_ID_COLUMN in reference_df.columns:
        reference_df[IMAGE_ID_COLUMN] = normalize_id_series(reference_df[IMAGE_ID_COLUMN])
    return reference_df, reference_path


def map_patient_ids_from_image_ids(image_ids, reference_df, reference_path, source_path):
    if IMAGE_ID_COLUMN not in reference_df.columns:
        raise ValueError(
            f"Cannot map image_id to patient_id because {reference_path} has no '{IMAGE_ID_COLUMN}' column."
        )

    reference_map = (
        reference_df[[IMAGE_ID_COLUMN, PATIENT_ID_COLUMN]]
        .drop_duplicates(subset=[IMAGE_ID_COLUMN])
        .set_index(IMAGE_ID_COLUMN)[PATIENT_ID_COLUMN]
    )
    patient_ids = normalize_id_series(image_ids).map(reference_map)
    missing = patient_ids.isna() | (patient_ids == "")
    if missing.any():
        missing_examples = normalize_id_series(image_ids)[missing].head(5).tolist()
        raise ValueError(
            f"Cannot map {int(missing.sum())} image_id values from {source_path} "
            f"to patient_id using {reference_path}. Examples: {missing_examples}"
        )
    return patient_ids.reset_index(drop=True)


def recover_patient_ids_from_sidecar(source_path, row_count, reference_df, reference_path):
    sidecar_candidates = [
        source_path.with_name("data_process.csv"),
        source_path.with_name("table_fts.csv"),
        source_path.with_name("image_fts_norm.csv"),
        source_path.with_name("images_ft.csv"),
    ]
    seen = set()

    for sidecar_path in sidecar_candidates:
        if sidecar_path in seen or not sidecar_path.exists() or sidecar_path == source_path:
            continue
        seen.add(sidecar_path)

        sidecar_df = pd.read_csv(
            sidecar_path,
            dtype={IMAGE_ID_COLUMN: str, PATIENT_ID_COLUMN: str, "id": str},
        )
        if len(sidecar_df) != row_count:
            continue
        if PATIENT_ID_COLUMN in sidecar_df.columns:
            patient_ids = normalize_id_series(sidecar_df[PATIENT_ID_COLUMN])
            if not (patient_ids == "").any():
                return patient_ids.reset_index(drop=True), f"{sidecar_path}:patient_id_row_order"
        if IMAGE_ID_COLUMN in sidecar_df.columns:
            return (
                map_patient_ids_from_image_ids(
                    sidecar_df[IMAGE_ID_COLUMN],
                    reference_df,
                    reference_path,
                    sidecar_path,
                ),
                f"{sidecar_path}:image_id_to_patient_id",
            )

    return None, None


def infer_patient_ids(df, source_path, label_col, patient_id_source):
    if PATIENT_ID_COLUMN in df.columns and PATIENT_ID_COLUMN != label_col:
        patient_ids = normalize_id_series(df[PATIENT_ID_COLUMN])
        if (patient_ids == "").any():
            raise ValueError(f"Source CSV contains empty patient_id values: {source_path}")
        return patient_ids.reset_index(drop=True), f"{source_path}:patient_id"

    reference_df, reference_path = load_patient_id_reference(patient_id_source)
    if IMAGE_ID_COLUMN in df.columns and IMAGE_ID_COLUMN != label_col:
        return (
            map_patient_ids_from_image_ids(
                df[IMAGE_ID_COLUMN],
                reference_df,
                reference_path,
                source_path,
            ),
            f"{source_path}:image_id_to_patient_id",
        )

    if "id" in df.columns and "id" != label_col:
        patient_ids = normalize_id_series(df["id"])
        if (patient_ids == "").any():
            raise ValueError(f"Source CSV contains empty id values: {source_path}")
        return patient_ids.reset_index(drop=True), f"{source_path}:id_as_patient_id"

    sidecar_ids, sidecar_source = recover_patient_ids_from_sidecar(
        source_path,
        len(df),
        reference_df,
        reference_path,
    )
    if sidecar_ids is not None:
        return sidecar_ids, sidecar_source

    if source_path.parent == reference_path.parent and len(reference_df) == len(df):
        return (
            reference_df[PATIENT_ID_COLUMN].reset_index(drop=True),
            f"{reference_path}:patient_id_row_order",
        )

    raise ValueError(
        f"Cannot create patient-level KFold for {source_path}. Add a '{PATIENT_ID_COLUMN}' "
        f"column, add an '{IMAGE_ID_COLUMN}' column that exists in {reference_path}, "
        "or pass --patient-id-source for the matching raw metadata CSV."
    )


def load_source_frame(source_path, patient_id_source):
    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")

    df = pd.read_csv(source_path, dtype={IMAGE_ID_COLUMN: str, PATIENT_ID_COLUMN: str, "id": str})
    label_col = LABEL_COLUMN if LABEL_COLUMN in df.columns else df.columns[-1]

    id_columns = [col for col in ID_COLUMNS if col in df.columns and col != label_col]
    id_frame = df[id_columns].copy() if id_columns else pd.DataFrame(index=df.index)
    patient_ids, patient_id_source_used = infer_patient_ids(
        df,
        source_path,
        label_col,
        patient_id_source,
    )
    id_frame[PATIENT_ID_COLUMN] = patient_ids.values
    feature_df = df.drop(columns=[label_col] + id_columns)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    raw_labels = df[label_col]
    label_encoder = LabelEncoder()
    encoded_labels = pd.Series(
        label_encoder.fit_transform(raw_labels.astype(str)),
        name=LABEL_COLUMN,
        index=df.index,
    )
    label_mapping = {
        str(encoded): str(original)
        for encoded, original in enumerate(label_encoder.classes_)
    }
    return feature_df, encoded_labels, id_frame.reset_index(drop=True), label_mapping, patient_id_source_used


def limit_rows(feature_df, labels, id_frame, max_rows, seed):
    if max_rows is None or max_rows >= len(labels):
        return feature_df, labels, id_frame
    if max_rows < labels.nunique():
        raise ValueError("--max-rows must be at least the number of label classes.")

    if id_frame is not None and PATIENT_ID_COLUMN in id_frame.columns:
        rng = np.random.default_rng(seed)
        group_frame = pd.DataFrame(
            {
                PATIENT_ID_COLUMN: normalize_id_series(id_frame[PATIENT_ID_COLUMN]),
                LABEL_COLUMN: labels.reset_index(drop=True),
            },
            index=labels.index,
        )
        group_labels = (
            group_frame.groupby(PATIENT_ID_COLUMN, sort=False)[LABEL_COLUMN]
            .agg(lambda values: values.value_counts().sort_index().idxmax())
            .reset_index()
        )
        selected_groups = []
        counts = group_labels[LABEL_COLUMN].value_counts().sort_index()
        base_per_class = max(1, max_rows // max(1, len(counts)))
        remaining_rows = max_rows

        for label_value, _count in counts.items():
            candidates = group_labels[group_labels[LABEL_COLUMN] == label_value][PATIENT_ID_COLUMN].to_numpy()
            rng.shuffle(candidates)
            taken_for_class = 0
            for group_id in candidates:
                if taken_for_class >= base_per_class and remaining_rows <= 0:
                    break
                if group_id in selected_groups:
                    continue
                selected_groups.append(group_id)
                remaining_rows -= int((group_frame[PATIENT_ID_COLUMN] == group_id).sum())
                taken_for_class += 1
                if remaining_rows <= 0 and len(selected_groups) >= len(counts):
                    break

        if len(selected_groups) < len(counts):
            raise ValueError("--max-rows selected too few patient groups for all label classes.")

        selected_set = set(selected_groups)
        selected_mask = group_frame[PATIENT_ID_COLUMN].isin(selected_set)
        selected_indexes = labels.index[selected_mask].tolist()
        selected_indexes = sorted(selected_indexes)
        return (
            feature_df.loc[selected_indexes].reset_index(drop=True),
            labels.loc[selected_indexes].reset_index(drop=True),
            id_frame.loc[selected_indexes].reset_index(drop=True),
        )

    sample_parts = []
    counts = labels.value_counts().sort_index()
    base_per_class = max_rows // len(counts)
    remaining = max_rows - base_per_class * len(counts)
    rng = np.random.default_rng(seed)

    for label_value, count in counts.items():
        take = min(int(count), base_per_class + (1 if remaining > 0 else 0))
        remaining = max(0, remaining - 1)
        label_indexes = labels[labels == label_value].index.to_numpy()
        chosen = rng.choice(label_indexes, size=take, replace=False)
        sample_parts.extend(chosen.tolist())

    if len(sample_parts) < max_rows:
        available = np.array([idx for idx in labels.index if idx not in set(sample_parts)])
        extra = rng.choice(available, size=max_rows - len(sample_parts), replace=False)
        sample_parts.extend(extra.tolist())

    sample_parts = sorted(sample_parts)
    limited_ids = id_frame.loc[sample_parts].reset_index(drop=True) if id_frame is not None else None
    return (
        feature_df.loc[sample_parts].reset_index(drop=True),
        labels.loc[sample_parts].reset_index(drop=True),
        limited_ids,
    )


def build_patient_group_splits(features, labels, ids, folds, seed):
    if ids is None or PATIENT_ID_COLUMN not in ids.columns:
        raise ValueError(f"Patient-level KFold requires a '{PATIENT_ID_COLUMN}' column.")

    groups = normalize_id_series(ids[PATIENT_ID_COLUMN]).reset_index(drop=True)
    if (groups == "").any():
        raise ValueError("Patient-level KFold received empty patient_id values.")

    group_count = int(groups.nunique())
    if group_count < folds:
        raise ValueError(
            f"Cannot create {folds} patient-level folds from only {group_count} unique patients."
        )

    splitters = []
    if StratifiedGroupKFold is not None:
        splitters.append(
            (
                "StratifiedGroupKFold",
                StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed),
            )
        )
    splitters.append(("GroupKFold", GroupKFold(n_splits=folds)))

    last_error = None
    for splitter_name, splitter in splitters:
        try:
            splits = list(splitter.split(features, labels, groups))
        except ValueError as exc:
            last_error = exc
            continue

        for train_index, test_index in splits:
            train_groups = set(groups.iloc[train_index].tolist())
            test_groups = set(groups.iloc[test_index].tolist())
            overlap = train_groups & test_groups
            if overlap:
                examples = sorted(overlap)[:5]
                raise RuntimeError(
                    f"{splitter_name} leaked patient_id values across train/test: {examples}"
                )
        return splitter_name, groups, splits

    if last_error is not None:
        raise ValueError(f"Cannot create patient-level folds: {last_error}") from last_error
    raise ValueError("Cannot create patient-level folds.")


def clean_numeric_split(train_df, test_df):
    medians = train_df.median(numeric_only=True).fillna(0)
    return train_df.fillna(medians), test_df.fillna(medians)


def scale_split(train_df, test_df, scaler):
    train_clean, test_clean = clean_numeric_split(train_df, test_df)
    train_scaled = scaler.fit_transform(train_clean)
    test_scaled = scaler.transform(test_clean)
    return (
        pd.DataFrame(train_scaled, columns=train_df.columns),
        pd.DataFrame(test_scaled, columns=test_df.columns),
    )


def selector_to_feature_rows(selector, source_columns, selected_columns, branch, output_start):
    scores = getattr(selector, "scores_", None)
    pvalues = getattr(selector, "pvalues_", None)
    selected_positions = {
        column: output_start + selected_columns.index(column)
        for column in selected_columns
    }

    rows = []
    for idx, column in enumerate(source_columns):
        rows.append(
            {
                "branch": branch,
                "selected": column in selected_positions,
                "source_column": str(column),
                "selected_output_column": (
                    selected_positions[column] if column in selected_positions else ""
                ),
                "score": None if scores is None else float(scores[idx]),
                "p_value": None if pvalues is None else float(pvalues[idx]),
            }
        )
    return rows


def select_k_best(train_df, test_df, labels, k, score_func, branch, output_start):
    if train_df.empty:
        raise ValueError(f"No feature columns available for branch '{branch}'.")
    actual_k = min(k, train_df.shape[1])
    if actual_k < 1:
        raise ValueError(f"k for branch '{branch}' must be at least 1.")

    selector = SelectKBest(score_func=score_func, k=actual_k)
    train_selected = selector.fit_transform(train_df, labels)
    test_selected = selector.transform(test_df)
    selected_columns = list(train_df.columns[selector.get_support()])
    feature_rows = selector_to_feature_rows(
        selector=selector,
        source_columns=list(train_df.columns),
        selected_columns=selected_columns,
        branch=branch,
        output_start=output_start,
    )
    return (
        pd.DataFrame(train_selected),
        pd.DataFrame(test_selected),
        selected_columns,
        feature_rows,
    )


def build_fis_frame(selected_features, labels):
    out = pd.DataFrame(selected_features).reset_index(drop=True)
    out.columns = [str(i) for i in range(out.shape[1])]
    out[LABEL_COLUMN] = pd.Series(labels).reset_index(drop=True)
    return out


def drop_train_constant_columns(train_selected, test_selected, feature_rows):
    train_df = pd.DataFrame(train_selected).reset_index(drop=True)
    test_df = pd.DataFrame(test_selected).reset_index(drop=True)
    keep_positions = [
        idx
        for idx in range(train_df.shape[1])
        if train_df.iloc[:, idx].nunique(dropna=False) > 1
    ]
    if len(keep_positions) == train_df.shape[1]:
        return train_df, test_df, feature_rows, []
    if not keep_positions:
        raise ValueError("All selected columns are constant on the train fold.")

    position_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_positions)}
    dropped_positions = [
        idx
        for idx in range(train_df.shape[1])
        if idx not in position_map
    ]
    updated_rows = []
    for row in feature_rows:
        updated = dict(row)
        selected_output = updated.get("selected_output_column", "")
        try:
            selected_output = int(float(selected_output))
        except (TypeError, ValueError):
            selected_output = None
        updated["dropped_constant_train"] = False
        if selected_output is not None:
            if selected_output in position_map:
                updated["selected_output_column"] = position_map[selected_output]
            else:
                updated["selected"] = False
                updated["selected_output_column"] = ""
                updated["dropped_constant_train"] = True
        updated_rows.append(updated)

    return (
        train_df.iloc[:, keep_positions].reset_index(drop=True),
        test_df.iloc[:, keep_positions].reset_index(drop=True),
        updated_rows,
        dropped_positions,
    )


def apply_train_smote(train_df, seed, max_k_neighbors):
    y = train_df[LABEL_COLUMN]
    x = train_df.drop(columns=[LABEL_COLUMN])
    counts = y.value_counts()
    min_count = int(counts.min())
    if min_count < 2:
        return train_df, "skipped_minority_class_too_small"

    k_neighbors = min(max_k_neighbors, min_count - 1)
    try:
        from imblearn.over_sampling import BorderlineSMOTE
    except ImportError:
        print("[WARN] imblearn is not installed; SMOTE skipped for this train fold.")
        return train_df, "skipped_imblearn_missing"

    sampler = BorderlineSMOTE(
        random_state=seed,
        sampling_strategy="auto",
        k_neighbors=k_neighbors,
    )
    x_resampled, y_resampled = sampler.fit_resample(x, y)
    resampled = pd.DataFrame(x_resampled, columns=x.columns)
    resampled[LABEL_COLUMN] = pd.Series(y_resampled).reset_index(drop=True)
    return resampled, f"borderline_smote_k_{k_neighbors}"


def prepare_image_fold(train_x, test_x, train_y, args, fold_seed):
    train_scaled, test_scaled = clean_numeric_split(train_x, test_x)
    train_selected, test_selected, selected_columns, rows = select_k_best(
        train_scaled,
        test_scaled,
        train_y,
        args.k_img,
        f_classif,
        branch="image",
        output_start=0,
    )
    cluster = [5] * len(selected_columns) + [2]
    return train_selected, test_selected, cluster, rows


def prepare_table_fold(train_x, test_x, train_y, args, fold_seed):
    train_scaled, test_scaled = scale_split(train_x, test_x, MinMaxScaler())
    score_func = partial(mutual_info_classif, random_state=fold_seed)
    train_selected, test_selected, selected_columns, rows = select_k_best(
        train_scaled,
        test_scaled,
        train_y,
        args.k_table,
        score_func,
        branch="table",
        output_start=0,
    )
    cluster = [TABLE_CLUSTER_BY_COLUMN.get(str(column), 5) for column in selected_columns] + [2]
    return train_selected, test_selected, cluster, rows


def fallback_fusion_split(columns):
    midpoint = len(columns) // 2
    return list(columns[:midpoint]), list(columns[midpoint:])


def split_fusion_columns(columns):
    table_columns = [col for col in TABLE_FEATURE_COLUMNS if col in columns]
    image_columns = [col for col in IMAGE_FEATURE_COLUMNS if col in columns]
    if not table_columns or not image_columns:
        table_columns, image_columns = fallback_fusion_split(list(columns))
    return table_columns, image_columns


def selected_index_feature_rows(source_columns, selected_indices, branch, output_start, scores=None):
    selected_indices = list(selected_indices)
    selected_positions = {
        idx: output_start + pos
        for pos, idx in enumerate(selected_indices)
    }
    rows = []
    for idx, column in enumerate(source_columns):
        rows.append(
            {
                "branch": branch,
                "selected": idx in selected_positions,
                "source_column": str(column),
                "selected_output_column": selected_positions[idx] if idx in selected_positions else "",
                "score": None if scores is None else float(scores[idx]),
                "p_value": None,
            }
        )
    return rows


def generated_component_feature_rows(method_name, feature_count):
    return [
        {
            "branch": method_name,
            "selected": True,
            "source_column": f"{method_name}_component_{idx}",
            "selected_output_column": idx,
            "score": None,
            "p_value": None,
        }
        for idx in range(feature_count)
    ]


def prepare_fusion_fold(train_x, test_x, train_y, args, fold_seed):
    table_columns, image_columns = split_fusion_columns(train_x.columns)

    train_img, test_img = scale_split(
        train_x[image_columns],
        test_x[image_columns],
        StandardScaler(),
    )
    train_tab, test_tab = scale_split(
        train_x[table_columns],
        test_x[table_columns],
        StandardScaler(),
    )

    train_img_selected, test_img_selected, img_columns, img_rows = select_k_best(
        train_img,
        test_img,
        train_y,
        args.k_img,
        f_classif,
        branch="image",
        output_start=0,
    )

    train_tab_selected, test_tab_selected, tab_columns, tab_rows = select_k_best(
        train_tab,
        test_tab,
        train_y,
        args.k_tab,
        f_classif,
        branch="table",
        output_start=len(img_columns),
    )

    train_selected = pd.concat([train_img_selected, train_tab_selected], axis=1, ignore_index=True)
    test_selected = pd.concat([test_img_selected, test_tab_selected], axis=1, ignore_index=True)

    # The legacy FT-selection fusion scenario uses [5] for every selected feature.
    cluster = [5] * (len(img_columns) + len(tab_columns)) + [2]
    return train_selected, test_selected, cluster, img_rows + tab_rows


def compute_filter_scores(train_values, train_y, seed):
    from sklearn.ensemble import RandomForestClassifier

    x_scaled = MinMaxScaler().fit_transform(train_values)
    mi_scores = mutual_info_classif(x_scaled, train_y, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(x_scaled, train_y)
    return (mi_scores + rf.feature_importances_) / 2


def remove_correlated_indices(train_values, indices, threshold):
    selected = []
    for idx in indices:
        too_correlated = False
        for selected_idx in selected:
            corr = np.corrcoef(train_values[:, idx], train_values[:, selected_idx])[0, 1]
            if np.isfinite(corr) and abs(corr) > threshold:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(int(idx))
    return selected


def prepare_fusion_filter_fold(train_x, test_x, train_y, args, fold_seed):
    table_columns, image_columns = split_fusion_columns(train_x.columns)
    train_img, test_img = scale_split(train_x[image_columns], test_x[image_columns], StandardScaler())
    train_tab, test_tab = scale_split(train_x[table_columns], test_x[table_columns], StandardScaler())

    img_values = train_img.to_numpy()
    tab_values = train_tab.to_numpy()
    img_scores = compute_filter_scores(img_values, train_y, fold_seed)
    tab_scores = compute_filter_scores(tab_values, train_y, fold_seed + 1)
    candidate_img = np.argsort(img_scores)[::-1][: 2 * args.k_img]
    candidate_tab = np.argsort(tab_scores)[::-1][: 2 * args.k_tab]
    selected_img = remove_correlated_indices(img_values, candidate_img, args.filter_corr)[: args.k_img]
    selected_tab = remove_correlated_indices(tab_values, candidate_tab, args.filter_corr)[: args.k_tab]

    train_selected = pd.concat(
        [train_img.iloc[:, selected_img], train_tab.iloc[:, selected_tab]],
        axis=1,
        ignore_index=True,
    )
    test_selected = pd.concat(
        [test_img.iloc[:, selected_img], test_tab.iloc[:, selected_tab]],
        axis=1,
        ignore_index=True,
    )
    rows = selected_index_feature_rows(image_columns, selected_img, "image_filter", 0, img_scores)
    rows += selected_index_feature_rows(table_columns, selected_tab, "table_filter", len(selected_img), tab_scores)
    cluster = [5] * len(selected_img) + [
        TABLE_CLUSTER_BY_COLUMN.get(str(table_columns[idx]), 5)
        for idx in selected_tab
    ] + [2]
    return train_selected, test_selected, cluster, rows


def normalize_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True) + 1e-8
    return values / norms


def prepare_fusion_hadamard_fold(train_x, test_x, train_y, args, fold_seed):
    from sklearn.linear_model import Ridge

    table_columns, image_columns = split_fusion_columns(train_x.columns)
    train_img, test_img = scale_split(train_x[image_columns], test_x[image_columns], StandardScaler())
    train_tab, test_tab = scale_split(train_x[table_columns], test_x[table_columns], StandardScaler())

    common_dim = int(args.hadamard_dim)
    if common_dim < 1:
        raise ValueError("--hadamard-dim must be at least 1.")
    rng = np.random.default_rng(fold_seed)
    x_random = rng.standard_normal((train_img.shape[0], common_dim))
    y_random = rng.standard_normal((train_tab.shape[0], common_dim))
    proj_img = Ridge(alpha=0.01, fit_intercept=False)
    proj_tab = Ridge(alpha=0.01, fit_intercept=False)
    proj_img.fit(train_img, x_random)
    proj_tab.fit(train_tab, y_random)

    train_img_proj = normalize_rows(proj_img.predict(train_img))
    train_tab_proj = normalize_rows(proj_tab.predict(train_tab))
    test_img_proj = normalize_rows(proj_img.predict(test_img))
    test_tab_proj = normalize_rows(proj_tab.predict(test_tab))
    train_selected = np.concatenate(
        [train_img_proj * train_tab_proj, np.tanh(train_img_proj), np.tanh(train_tab_proj)],
        axis=1,
    )
    test_selected = np.concatenate(
        [test_img_proj * test_tab_proj, np.tanh(test_img_proj), np.tanh(test_tab_proj)],
        axis=1,
    )
    feature_count = train_selected.shape[1]
    cluster = [5] * feature_count + [2]
    return (
        pd.DataFrame(train_selected),
        pd.DataFrame(test_selected),
        cluster,
        generated_component_feature_rows("hadamard", feature_count),
    )


def outer_product_features(left_values, right_values):
    return np.einsum("ij,ik->ijk", left_values, right_values).reshape(left_values.shape[0], -1)


def prepare_fusion_tensor_fold(train_x, test_x, train_y, args, fold_seed):
    from sklearn.decomposition import TruncatedSVD

    table_columns, image_columns = split_fusion_columns(train_x.columns)
    train_img, test_img = scale_split(train_x[image_columns], test_x[image_columns], StandardScaler())
    train_tab, test_tab = scale_split(train_x[table_columns], test_x[table_columns], StandardScaler())

    train_outer = outer_product_features(train_img.to_numpy(), train_tab.to_numpy())
    test_outer = outer_product_features(test_img.to_numpy(), test_tab.to_numpy())
    rank = min(int(args.tensor_rank), train_outer.shape[1])
    if rank < 1:
        raise ValueError("--tensor-rank must be at least 1.")
    svd = TruncatedSVD(n_components=rank, random_state=fold_seed)
    train_selected = svd.fit_transform(train_outer)
    test_selected = svd.transform(test_outer)
    cluster = [5] * rank + [2]
    rows = generated_component_feature_rows("tensor", rank)
    return pd.DataFrame(train_selected), pd.DataFrame(test_selected), cluster, rows


def evaluate_wrapper_feature_set(values, target, seed, cv, estimators):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    label_counts = pd.Series(target).value_counts()
    actual_cv = min(int(cv), int(label_counts.min()))
    if actual_cv < 2:
        model = RandomForestClassifier(n_estimators=estimators, random_state=seed, n_jobs=-1)
        model.fit(values, target)
        return float(model.score(values, target))
    model = RandomForestClassifier(n_estimators=estimators, random_state=seed, n_jobs=-1)
    scores = cross_val_score(model, values, target, cv=actual_cv, scoring="accuracy")
    return float(scores.mean())


def best_wrapper_feature(values, selected_indices, target, remaining_indices, seed, cv, estimators, cache):
    best_score = -np.inf
    best_feature = None
    for idx in sorted(remaining_indices):
        trial = selected_indices + [idx]
        cache_key = ("single", tuple(trial))
        if cache_key not in cache:
            cache[cache_key] = evaluate_wrapper_feature_set(values[:, trial], target, seed, cv, estimators)
        score = cache[cache_key]
        if score > best_score:
            best_score = score
            best_feature = idx
    return best_feature, best_score


def select_wrapper_indices(img_values, tab_values, target, args, seed):
    selected_img = []
    selected_tab = []
    img_indices = set(range(img_values.shape[1]))
    tab_indices = set(range(tab_values.shape[1]))
    score_cache = {}
    cv = max(2, int(args.wrapper_cv))
    estimators = max(1, int(args.wrapper_rf_estimators))

    for _ in range(min(args.wrapper_min_img, args.wrapper_max_img, img_values.shape[1])):
        best_feature, _ = best_wrapper_feature(
            img_values,
            selected_img,
            target,
            img_indices - set(selected_img),
            seed,
            cv,
            estimators,
            score_cache,
        )
        if best_feature is not None:
            selected_img.append(best_feature)

    for _ in range(min(args.wrapper_min_tab, args.wrapper_max_tab, tab_values.shape[1])):
        best_feature, _ = best_wrapper_feature(
            tab_values,
            selected_tab,
            target,
            tab_indices - set(selected_tab),
            seed,
            cv,
            estimators,
            score_cache,
        )
        if best_feature is not None:
            selected_tab.append(best_feature)

    fused = np.concatenate([img_values[:, selected_img], tab_values[:, selected_tab]], axis=1)
    best_score = evaluate_wrapper_feature_set(fused, target, seed, cv, estimators)

    while len(selected_img) < min(args.wrapper_max_img, img_values.shape[1]) or len(selected_tab) < min(args.wrapper_max_tab, tab_values.shape[1]):
        best_new_score = -np.inf
        best_new_feature = None
        best_modality = None

        if len(selected_img) < min(args.wrapper_max_img, img_values.shape[1]):
            for idx in sorted(img_indices - set(selected_img)):
                fused_trial = np.concatenate(
                    [img_values[:, selected_img + [idx]], tab_values[:, selected_tab]],
                    axis=1,
                )
                cache_key = ("image", tuple(selected_img + [idx]), tuple(selected_tab))
                if cache_key not in score_cache:
                    score_cache[cache_key] = evaluate_wrapper_feature_set(
                        fused_trial,
                        target,
                        seed,
                        cv,
                        estimators,
                    )
                score = score_cache[cache_key]
                if score > best_new_score:
                    best_new_score = score
                    best_new_feature = idx
                    best_modality = "image"

        if len(selected_tab) < min(args.wrapper_max_tab, tab_values.shape[1]):
            for idx in sorted(tab_indices - set(selected_tab)):
                fused_trial = np.concatenate(
                    [img_values[:, selected_img], tab_values[:, selected_tab + [idx]]],
                    axis=1,
                )
                cache_key = ("table", tuple(selected_img), tuple(selected_tab + [idx]))
                if cache_key not in score_cache:
                    score_cache[cache_key] = evaluate_wrapper_feature_set(
                        fused_trial,
                        target,
                        seed,
                        cv,
                        estimators,
                    )
                score = score_cache[cache_key]
                if score > best_new_score:
                    best_new_score = score
                    best_new_feature = idx
                    best_modality = "table"

        if best_new_feature is None or best_new_score <= best_score:
            break
        best_score = best_new_score
        if best_modality == "image":
            selected_img.append(best_new_feature)
        else:
            selected_tab.append(best_new_feature)

    return selected_img, selected_tab


def prepare_fusion_wrapper_fold(train_x, test_x, train_y, args, fold_seed):
    table_columns, image_columns = split_fusion_columns(train_x.columns)
    train_img, test_img = scale_split(train_x[image_columns], test_x[image_columns], StandardScaler())
    train_tab, test_tab = scale_split(train_x[table_columns], test_x[table_columns], StandardScaler())
    img_values = train_img.to_numpy()
    tab_values = train_tab.to_numpy()
    selected_img, selected_tab = select_wrapper_indices(img_values, tab_values, train_y.to_numpy(), args, fold_seed)

    train_selected = pd.concat(
        [train_img.iloc[:, selected_img], train_tab.iloc[:, selected_tab]],
        axis=1,
        ignore_index=True,
    )
    test_selected = pd.concat(
        [test_img.iloc[:, selected_img], test_tab.iloc[:, selected_tab]],
        axis=1,
        ignore_index=True,
    )
    rows = selected_index_feature_rows(image_columns, selected_img, "image_wrapper", 0)
    rows += selected_index_feature_rows(table_columns, selected_tab, "table_wrapper", len(selected_img))
    cluster = [5] * len(selected_img) + [
        TABLE_CLUSTER_BY_COLUMN.get(str(table_columns[idx]), 5)
        for idx in selected_tab
    ] + [2]
    return train_selected, test_selected, cluster, rows


def prepare_fusion_variant_fold(config_key, train_x, test_x, train_y, args, fold_seed):
    if config_key == "fusion_filter":
        return prepare_fusion_filter_fold(train_x, test_x, train_y, args, fold_seed)
    if config_key == "fusion_hadamard":
        return prepare_fusion_hadamard_fold(train_x, test_x, train_y, args, fold_seed)
    if config_key == "fusion_tensor":
        return prepare_fusion_tensor_fold(train_x, test_x, train_y, args, fold_seed)
    if config_key == "fusion_wrapper":
        return prepare_fusion_wrapper_fold(train_x, test_x, train_y, args, fold_seed)
    raise ValueError(f"Unsupported fusion variant: {config_key}")


def write_heatmap(df, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        df_features = df.iloc[:, :-1]
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_features.corr(), annot=False, cmap="coolwarm", fmt=".2f")
        plt.title("Heatmap")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Heatmap skipped: {exc}")


def save_confusion_plot(true_labels, predicted_labels, labels, output_path, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(values_format="d")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Confusion matrix image skipped: {exc}")


def save_score_plot(metrics, output_path, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [name for name in ["accuracy", "precision", "recall", "specificity", "f1", "auc"] if name in metrics]
        values = [metrics[name] for name in names]
        plot_values = [0.0 if pd.isna(value) else value for value in values]
        colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]
        plt.figure(figsize=(9, 5))
        bars = plt.bar([name.title() for name in names], plot_values, color=colors[: len(names)])
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title(title)
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                min((0.0 if pd.isna(value) else value) + 0.02, 1.03),
                "NA" if pd.isna(value) else f"{value:.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Score image skipped: {exc}")


def evaluate_fis_direct(file_name, test_df, output_dir, modality):
    from module.Test_FIS.fuzzify_input import fuzzify_input
    from module.Test_FIS.match_rule import match_rule
    from models.load_model import load_model
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    start_time = time.time()
    model_data = load_model(fileName=file_name)
    rule_list = np.array(model_data["ruleList"])
    sigma_m = np.array(model_data["sigma_M"]).flatten()
    centers = np.array(model_data["centers"], dtype=object)

    true_labels = []
    predicted_labels = []
    label_index = test_df.shape[1] - 1
    for _, row in test_df.iterrows():
        sample_input = row.values[:label_index]
        fuzzy_input = fuzzify_input(sample_input, sigma_m, centers)
        predicted_label = match_rule(fuzzy_input, rule_list)
        predicted_labels.append(0 if predicted_label is None else int(predicted_label))
        true_labels.append(int(float(row.values[label_index])) + 1)

    labels = sorted(set(true_labels) | set(predicted_labels))
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "f1": f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
    }
    elapsed = time.time() - start_time

    predictions_path = output_dir / "Predictions_FIS.csv"
    pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_label": predicted_labels,
        }
    ).to_csv(predictions_path, index=False)

    results_path = output_dir / "Results_FIS.csv"
    pd.DataFrame(
        {
            "Total Time": [elapsed],
            "Test Accuracy": [metrics["accuracy"]],
            "Test Precision": [metrics["precision"]],
            "Test Recall": [metrics["recall"]],
            "Test F1": [metrics["f1"]],
        }
    ).to_csv(results_path, index=False)

    conf_matrix_path = output_dir / "conf_matrix_fis.png"
    scores_path = output_dir / "scores_fis.png"
    save_confusion_plot(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        labels=labels,
        output_path=conf_matrix_path,
        title=f"FIS Confusion Matrix\n{modality}",
    )
    save_score_plot(
        metrics=metrics,
        output_path=scores_path,
        title=f"FIS Scores\n{modality}",
    )

    return {
        "fis_eval_time_seconds": elapsed,
        "fis_accuracy": metrics["accuracy"],
        "fis_precision": metrics["precision"],
        "fis_recall": metrics["recall"],
        "fis_f1": metrics["f1"],
        "fis_results_csv": str(results_path),
        "fis_predictions_csv": str(predictions_path),
        "fis_confusion_matrix_png": str(conf_matrix_path),
        "fis_scores_png": str(scores_path),
    }


def get_language_converter():
    try:
        from module.Convert import var_lang

        converter_name = next(
            name
            for name in dir(var_lang)
            if name.startswith("change_var_lang") and name.endswith("_default")
        )
        return getattr(var_lang, converter_name)
    except Exception:
        return None


def run_legacy_fis_for_split(
    file_name,
    train_df,
    test_df,
    cluster,
    modality,
    range_source,
    write_fold_heatmap,
    run_fis_test,
):
    from module.FIS.FIS import Generator_rule_with_data
    from module.Rules_Function.RuleWeight import RuleWeight
    from module.Rules_Function.Rules_gen import rule_generate
    from module.Rules_Function.Rules_reduce import reduce_rule, remove_rule

    import pickle

    os.chdir(PROJECT_ROOT)
    base_dir = PROJECT_ROOT
    input_dir = base_dir / "data" / "FIS" / "input" / Path(file_name)
    output_dir = base_dir / "data" / "FIS" / "output" / Path(file_name)
    output_dir_frb = output_dir / "FRB"
    model_dir = base_dir / "models" / Path(file_name)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_frb.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(input_dir / "train_data.csv", index=False)
    test_df.to_csv(input_dir / "test_data.csv", index=False)

    if write_fold_heatmap:
        write_heatmap(pd.concat([train_df, test_df], ignore_index=True), input_dir / "heatmap.png")

    start_time = time.time()
    train_data = np.array(train_df)
    test_data = np.array(test_df)
    full_data = np.array(pd.concat([train_df, test_df], ignore_index=True))
    range_data = train_data if range_source == "train" else full_data

    min_vals = np.min(range_data, axis=0)
    max_vals = np.max(range_data, axis=0)
    pd.DataFrame(min_vals).to_csv(output_dir / "min_vals.csv", index=False)
    pd.DataFrame(max_vals).to_csv(output_dir / "max_vals.csv", index=False)

    h, w = train_data.shape
    if len(cluster) != w:
        raise ValueError(
            f"Cluster length mismatch for {file_name}: len(cluster)={len(cluster)}, columns={w}"
        )

    m = 2
    esp = 0.01
    max_test = 200
    rules, centers, u_matrix = rule_generate(
        h,
        w,
        train_data,
        cluster,
        min_vals,
        max_vals,
        m,
        esp,
        max_test,
    )

    label_column_index = train_data.shape[1] - 1
    for row_idx in range(h):
        rules[row_idx, label_column_index] = np.argmax(u_matrix[row_idx, :]) + 1

    weights, sigma_m = RuleWeight(rules, train_data[:, :-1], cluster, centers)
    sigma_m = sigma_m.reshape(-1, 1)
    sigma_m = sigma_m[:-1, :]
    sigma_m = np.hstack((sigma_m[:, [0]], sigma_m[:, [0]], sigma_m[:, [0]]))

    df_rule_list_all = pd.DataFrame(rules)
    df_rule_list_all.to_csv(output_dir / "Rule_List_All.csv", index=False)

    rules_with_weight = np.hstack(
        (rules, np.min(weights, axis=1, keepdims=True), train_data[:, [label_column_index]])
    )
    rules_reduce = reduce_rule(h, label_column_index, rules_with_weight)
    pd.DataFrame(rules_reduce).to_csv(output_dir / "Rule_List_reduce.csv", index=False)

    rule_list_model = remove_rule(h, label_column_index, rules_reduce)
    rule_list = np.array(df_rule_list_all)

    converter = get_language_converter()
    if converter is not None:
        try:
            rule_list_language = converter(cluster, rule_list)
            pd.DataFrame(rule_list_language).to_csv(
                output_dir / "Rule_List_Language.csv",
                index=False,
            )
            pd.DataFrame(rule_list_language).to_csv(output_dir / "FRB.csv", index=False)
        except Exception as exc:
            print(f"[WARN] Language rule export skipped for {file_name}: {exc}")

    pd.DataFrame(rule_list).to_csv(output_dir / "Rule_List.csv", index=False)
    pd.DataFrame(sigma_m).to_csv(output_dir / "Sigma_M.csv", index=False)
    pd.DataFrame(centers).to_csv(output_dir / "Centers.csv", index=False)

    model_data = {
        "ruleList": rule_list_model,
        "sigma_M": sigma_m,
        "centers": centers,
        "min_vals": min_vals,
        "max_vals": max_vals,
    }
    with open(model_dir / "fuzzy_model.pkl", "wb") as file:
        pickle.dump(model_data, file)

    train_rule_df = df_rule_list_all
    test_rule_df = Generator_rule_with_data(
        data=pd.DataFrame(test_data),
        model_file=file_name,
    )
    test_rule_df.to_csv(output_dir_frb / "TestDataRule.csv", index=False)
    train_rule_df.to_csv(output_dir_frb / "TrainDataRule.csv", index=False)

    train_time = time.time() - start_time
    test_time = None
    fis_metrics = {}
    if run_fis_test:
        test_start = time.time()
        fis_metrics = evaluate_fis_direct(
            file_name=file_name,
            test_df=test_df,
            output_dir=output_dir,
            modality=modality,
        )
        test_time = time.time() - test_start

    result = {
        "fis_engine": "legacy",
        "fis_backend": "python",
        "file_name": file_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train_rule_path": str(output_dir_frb / "TrainDataRule.csv"),
        "test_rule_path": str(output_dir_frb / "TestDataRule.csv"),
        "model_dir": str(model_dir),
        "train_time_seconds": train_time,
        "fis_test_time_seconds": test_time,
    }
    result.update(fis_metrics)
    return result


def native_label_index_maps(train_labels, test_labels):
    labels = list(train_labels) + list(test_labels)
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx + 1 for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx + 1: label for idx, label in enumerate(unique_labels)}
    return label_to_idx, idx_to_label


def gauss_mf(value, sigma, center):
    if sigma <= 0:
        sigma = 1e-10
    diff = float(value) - float(center)
    return np.exp(-(diff * diff) / (2.0 * sigma * sigma))


def fuzzify_inputs_with_native_fis(fis, inputs):
    centers = fis.get_centers()
    sigma = fis.get_sigma()
    fuzzy_records = []
    for row in inputs:
        fuzzy_row = []
        for feature_idx, value in enumerate(row):
            center_vector = centers[feature_idx]
            feature_sigma = float(sigma[feature_idx])
            memberships = [
                gauss_mf(value, feature_sigma, center)
                for center in center_vector
            ]
            fuzzy_row.append(int(max(range(len(memberships)), key=memberships.__getitem__)) + 1)
        fuzzy_records.append(fuzzy_row)
    return fuzzy_records


def make_native_fis_rule_records(fis, train_inputs, train_labels, test_inputs, test_labels):
    label_to_idx, idx_to_label = native_label_index_maps(train_labels, test_labels)
    train_fuzzy = fuzzify_inputs_with_native_fis(fis, train_inputs)
    test_fuzzy = fuzzify_inputs_with_native_fis(fis, test_inputs)
    train_records = [
        fuzzy_row + [int(label_to_idx[label])]
        for fuzzy_row, label in zip(train_fuzzy, train_labels)
    ]
    test_records = [
        fuzzy_row + [int(label_to_idx[label])]
        for fuzzy_row, label in zip(test_fuzzy, test_labels)
    ]
    return train_records, test_records, label_to_idx, idx_to_label


def remap_native_fis_predictions(train_pred_clusters, train_true_labels, test_pred_clusters):
    from collections import Counter, defaultdict

    cluster_votes = defaultdict(Counter)
    for cluster_id, true_label in zip(train_pred_clusters, train_true_labels):
        cluster_votes[int(cluster_id)][true_label] += 1

    global_majority = Counter(train_true_labels).most_common(1)[0][0] if train_true_labels else 0
    cluster_to_label = {
        cluster_id: votes.most_common(1)[0][0]
        for cluster_id, votes in cluster_votes.items()
    }
    return [
        cluster_to_label.get(int(cluster_id), global_majority)
        for cluster_id in test_pred_clusters
    ]


def import_native_fisa_module():
    repo_root = PROJECT_ROOT.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from modules.fkg_python.fkg_runtime.module_loader import import_fisa_module

    return import_fisa_module(preferred="source", clear_existing=True)


def set_native_fis_backend(fisa_module, fis, requested_backend):
    gpu_compiled = bool(getattr(fisa_module, "GPU_COMPILED", False))
    gpu_available = bool(
        fisa_module.is_gpu_available()
        if hasattr(fisa_module, "is_gpu_available")
        else False
    )
    use_gpu = requested_backend == "gpu" or (
        requested_backend == "auto" and gpu_compiled and gpu_available
    )
    if hasattr(fis, "set_use_gpu"):
        fis.set_use_gpu(use_gpu)
    return "gpu" if use_gpu else "cpu", gpu_compiled, gpu_available


def run_native_fis_for_split(
    file_name,
    train_df,
    test_df,
    cluster,
    modality,
    range_source,
    write_fold_heatmap,
    run_fis_test,
    native_backend,
):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    os.chdir(PROJECT_ROOT)
    base_dir = PROJECT_ROOT
    input_dir = base_dir / "data" / "FIS" / "input" / Path(file_name)
    output_dir = base_dir / "data" / "FIS" / "output" / Path(file_name)
    output_dir_frb = output_dir / "FRB"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_frb.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(input_dir / "train_data.csv", index=False)
    test_df.to_csv(input_dir / "test_data.csv", index=False)
    if write_fold_heatmap:
        write_heatmap(pd.concat([train_df, test_df], ignore_index=True), input_dir / "heatmap.png")

    if len(cluster) != train_df.shape[1]:
        raise ValueError(
            f"Cluster length mismatch for {file_name}: len(cluster)={len(cluster)}, "
            f"columns={train_df.shape[1]}"
        )

    range_data = train_df if range_source == "train" else pd.concat([train_df, test_df], ignore_index=True)
    pd.DataFrame(range_data.min(axis=0)).to_csv(output_dir / "min_vals.csv", index=False)
    pd.DataFrame(range_data.max(axis=0)).to_csv(output_dir / "max_vals.csv", index=False)

    fisa_module, module_dir = import_native_fisa_module()
    fis = fisa_module.fis.FIS([int(value) for value in cluster], 2.0, 1e-5, 200)
    backend_used, gpu_compiled, gpu_available = set_native_fis_backend(
        fisa_module,
        fis,
        native_backend,
    )

    train_matrix = train_df.astype(float).values.tolist()
    train_inputs = train_df.iloc[:, :-1].astype(float).values.tolist()
    test_inputs = test_df.iloc[:, :-1].astype(float).values.tolist()
    train_labels = [int(float(v)) for v in train_df.iloc[:, -1].tolist()]
    test_labels = [int(float(v)) for v in test_df.iloc[:, -1].tolist()]

    train_start = time.perf_counter()
    fis.train(train_matrix)
    train_time = time.perf_counter() - train_start

    centers = fis.get_centers()
    sigma = fis.get_sigma()
    pd.DataFrame(centers).to_csv(output_dir / "Centers.csv", index=False)
    pd.DataFrame(sigma).to_csv(output_dir / "Sigma_M.csv", index=False)
    if hasattr(fis, "get_rules"):
        try:
            pd.DataFrame(fis.get_rules()).to_csv(output_dir / "Native_Rules.csv", index=False)
        except Exception as exc:
            print(f"[WARN] Native rule export skipped for {file_name}: {exc}")

    train_rule_records, test_rule_records, _label_to_idx, _idx_to_label = make_native_fis_rule_records(
        fis,
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
    )
    train_rule_df = pd.DataFrame(train_rule_records)
    test_rule_df = pd.DataFrame(test_rule_records)
    train_rule_df.to_csv(output_dir_frb / "TrainDataRule.csv", index=False)
    test_rule_df.to_csv(output_dir_frb / "TestDataRule.csv", index=False)
    train_rule_df.to_csv(output_dir / "Rule_List.csv", index=False)
    train_rule_df.to_csv(output_dir / "Rule_List_All.csv", index=False)

    fis_metrics = {}
    eval_time = None
    if run_fis_test:
        eval_start = time.perf_counter()
        train_pred_clusters = [int(value) for value in fis.predict_batch(train_inputs)]
        test_pred_clusters = [int(value) for value in fis.predict_batch(test_inputs)]
        predicted_labels = remap_native_fis_predictions(
            train_pred_clusters,
            train_labels,
            test_pred_clusters,
        )
        eval_time = time.perf_counter() - eval_start
        labels = sorted(set(test_labels) | set(predicted_labels))
        metrics = {
            "accuracy": accuracy_score(test_labels, predicted_labels),
            "precision": precision_score(test_labels, predicted_labels, average="macro", zero_division=0),
            "recall": recall_score(test_labels, predicted_labels, average="macro", zero_division=0),
            "f1": f1_score(test_labels, predicted_labels, average="macro", zero_division=0),
        }

        predictions_path = output_dir / "Predictions_FIS.csv"
        pd.DataFrame(
            {
                "true_label": test_labels,
                "predicted_label": predicted_labels,
                "predicted_cluster": test_pred_clusters,
            }
        ).to_csv(predictions_path, index=False)

        results_path = output_dir / "Results_FIS.csv"
        pd.DataFrame(
            {
                "Total Time": [eval_time],
                "Test Accuracy": [metrics["accuracy"]],
                "Test Precision": [metrics["precision"]],
                "Test Recall": [metrics["recall"]],
                "Test F1": [metrics["f1"]],
            }
        ).to_csv(results_path, index=False)

        conf_matrix_path = output_dir / "conf_matrix_fis.png"
        scores_path = output_dir / "scores_fis.png"
        save_confusion_plot(
            true_labels=test_labels,
            predicted_labels=predicted_labels,
            labels=labels,
            output_path=conf_matrix_path,
            title=f"FIS Confusion Matrix\n{modality}",
        )
        save_score_plot(
            metrics=metrics,
            output_path=scores_path,
            title=f"FIS Scores\n{modality}",
        )
        fis_metrics = {
            "fis_eval_time_seconds": eval_time,
            "fis_accuracy": metrics["accuracy"],
            "fis_precision": metrics["precision"],
            "fis_recall": metrics["recall"],
            "fis_f1": metrics["f1"],
            "fis_results_csv": str(results_path),
            "fis_predictions_csv": str(predictions_path),
            "fis_confusion_matrix_png": str(conf_matrix_path),
            "fis_scores_png": str(scores_path),
        }

    result = {
        "fis_engine": "native",
        "fis_backend": backend_used,
        "native_module_dir": str(module_dir),
        "gpu_compiled": gpu_compiled,
        "gpu_available": gpu_available,
        "file_name": file_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train_rule_path": str(output_dir_frb / "TrainDataRule.csv"),
        "test_rule_path": str(output_dir_frb / "TestDataRule.csv"),
        "model_dir": "",
        "train_time_seconds": train_time,
        "fis_test_time_seconds": eval_time,
    }
    result.update(fis_metrics)
    return result


def run_fis_for_split(
    file_name,
    train_df,
    test_df,
    cluster,
    modality,
    range_source,
    write_fold_heatmap,
    run_fis_test,
    fis_engine,
    native_backend,
):
    if fis_engine == "legacy":
        return run_legacy_fis_for_split(
            file_name=file_name,
            train_df=train_df,
            test_df=test_df,
            cluster=cluster,
            modality=modality,
            range_source=range_source,
            write_fold_heatmap=write_fold_heatmap,
            run_fis_test=run_fis_test,
        )
    return run_native_fis_for_split(
        file_name=file_name,
        train_df=train_df,
        test_df=test_df,
        cluster=cluster,
        modality=modality,
        range_source=range_source,
        write_fold_heatmap=write_fold_heatmap,
        run_fis_test=run_fis_test,
        native_backend=native_backend,
    )


def set_native_fkg_backend(fisa_module, fkg, requested_backend):
    gpu_compiled = bool(getattr(fisa_module, "GPU_COMPILED", False))
    gpu_available = bool(
        fisa_module.is_gpu_available()
        if hasattr(fisa_module, "is_gpu_available")
        else False
    )
    use_gpu = requested_backend == "gpu" or (
        requested_backend == "auto" and gpu_compiled and gpu_available
    )
    if hasattr(fkg, "set_use_gpu"):
        fkg.set_use_gpu(use_gpu)
    return "gpu" if use_gpu else "cpu", gpu_compiled, gpu_available


def save_confusion_csv(true_labels, predicted_labels, labels, output_path):
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    out = pd.DataFrame(matrix, index=[f"true_{label}" for label in labels])
    out.columns = [f"pred_{label}" for label in labels]
    out.to_csv(output_path)


def binary_auc_score(true_labels, positive_scores, positive_label):
    positives = [1 if value == positive_label else 0 for value in true_labels]
    positive_count = sum(positives)
    negative_count = len(positives) - positive_count
    if positive_count == 0 or negative_count == 0:
        return math.nan

    order = sorted(range(len(positive_scores)), key=lambda index: positive_scores[index])
    ranks = [0.0] * len(positive_scores)
    cursor = 0
    while cursor < len(order):
        next_cursor = cursor + 1
        while (
            next_cursor < len(order)
            and positive_scores[order[next_cursor]] == positive_scores[order[cursor]]
        ):
            next_cursor += 1
        average_rank = (cursor + 1 + next_cursor) / 2.0
        for rank_index in range(cursor, next_cursor):
            ranks[order[rank_index]] = average_rank
        cursor = next_cursor

    positive_rank_sum = sum(rank for rank, is_positive in zip(ranks, positives) if is_positive)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def binary_specificity(true_labels, predicted_labels, positive_label):
    tn = sum(
        1
        for truth, predicted in zip(true_labels, predicted_labels)
        if truth != positive_label and predicted != positive_label
    )
    fp = sum(
        1
        for truth, predicted in zip(true_labels, predicted_labels)
        if truth != positive_label and predicted == positive_label
    )
    return tn / (tn + fp) if (tn + fp) else 0.0


def positive_scores_from_predictions(predicted_labels, confidences, positive_label):
    scores = []
    for predicted, confidence in zip(predicted_labels, confidences):
        if confidence is None or pd.isna(confidence) or not np.isfinite(confidence):
            scores.append(1.0 if predicted == positive_label else 0.0)
            continue
        confidence = max(0.0, min(1.0, float(confidence)))
        scores.append(confidence if predicted == positive_label else 1.0 - confidence)
    return scores


def run_native_fkg_for_rules(train_rule_path, test_rule_path, output_dir, modality, backend):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    train_df = pd.read_csv(train_rule_path)
    test_df = pd.read_csv(test_rule_path)
    train_records = [[int(float(value)) for value in row] for row in train_df.values.tolist()]
    test_records = [[int(float(value)) for value in row] for row in test_df.values.tolist()]
    if not train_records or not test_records:
        raise ValueError(f"FKG rule data is empty for {modality}.")

    train_labels = [int(row[-1]) for row in train_records]
    test_labels = [int(row[-1]) for row in test_records]
    n_classes = max(max(train_labels + test_labels), len(set(train_labels + test_labels)))

    fisa_module, module_dir = import_native_fisa_module()
    fkg = fisa_module.fkg.FKG()
    backend_used, gpu_compiled, gpu_available = set_native_fkg_backend(
        fisa_module,
        fkg,
        backend,
    )

    train_start = time.perf_counter()
    fkg.train(train_records, int(n_classes))
    train_time = time.perf_counter() - train_start

    test_inputs = [row[:-1] for row in test_records]
    test_start = time.perf_counter()
    predicted_labels = []
    confidences = []
    if hasattr(fkg, "predict_batch_with_confidence"):
        for pred, confidence in fkg.predict_batch_with_confidence(test_inputs):
            predicted_labels.append(int(pred))
            confidences.append(float(confidence))
    elif hasattr(fkg, "predict_batch"):
        predicted_labels = [int(value) for value in fkg.predict_batch(test_inputs)]
        confidences = [None] * len(predicted_labels)
    else:
        for sample in test_inputs:
            pred, confidence = fkg.predict(sample)
            predicted_labels.append(int(pred))
            confidences.append(float(confidence))
    test_time = time.perf_counter() - test_start

    labels = sorted(set(test_labels) | set(predicted_labels))
    positive_label = max(labels)
    positive_scores = positive_scores_from_predictions(predicted_labels, confidences, positive_label)
    metrics = {
        "accuracy": accuracy_score(test_labels, predicted_labels),
        "precision": precision_score(test_labels, predicted_labels, average="macro", zero_division=0),
        "recall": recall_score(test_labels, predicted_labels, average="macro", zero_division=0),
        "specificity": binary_specificity(test_labels, predicted_labels, positive_label),
        "f1": f1_score(test_labels, predicted_labels, average="macro", zero_division=0),
        "auc": binary_auc_score(test_labels, positive_scores, positive_label),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "Predictions_FKG.csv"
    results_path = output_dir / "Results_FKG.csv"
    conf_matrix_csv = output_dir / "conf_matrix_fkg.csv"
    conf_matrix_path = output_dir / "conf_matrix_fkg.png"
    scores_path = output_dir / "scores_fkg.png"

    pd.DataFrame(
        {
            "true_label": test_labels,
            "predicted_label": predicted_labels,
            "confidence": confidences,
            "positive_score": positive_scores,
        }
    ).to_csv(predictions_path, index=False)

    pd.DataFrame(
        {
            "Train Time": [train_time],
            "Test Time": [test_time],
            "Total Time": [train_time + test_time],
            "Test Accuracy": [metrics["accuracy"]],
            "Test Precision": [metrics["precision"]],
            "Test Recall": [metrics["recall"]],
            "Test Specificity": [metrics["specificity"]],
            "Test F1": [metrics["f1"]],
            "Test AUC": [metrics["auc"]],
            "Positive Label": [positive_label],
            "Engine": ["native_fisa_module"],
            "Backend Request": [backend],
            "Backend Used": [backend_used],
            "GPU Compiled": [gpu_compiled],
            "GPU Available": [gpu_available],
            "Train Samples": [len(train_records)],
            "Test Samples": [len(test_records)],
            "Feature Count": [len(train_records[0]) - 1],
            "Module Dir": [str(module_dir)],
        }
    ).to_csv(results_path, index=False)

    save_confusion_csv(test_labels, predicted_labels, labels, conf_matrix_csv)
    save_confusion_plot(
        true_labels=test_labels,
        predicted_labels=predicted_labels,
        labels=labels,
        output_path=conf_matrix_path,
        title=f"FKG Confusion Matrix\n{modality}",
    )
    save_score_plot(
        metrics=metrics,
        output_path=scores_path,
        title=f"FKG Scores\n{modality}",
    )

    return {
        "fkg_engine": "native_fisa_module",
        "fkg_backend_request": backend,
        "fkg_backend": backend_used,
        "fkg_native_module_dir": str(module_dir),
        "fkg_gpu_compiled": gpu_compiled,
        "fkg_gpu_available": gpu_available,
        "fkg_train_time_seconds": train_time,
        "fkg_test_time_seconds": test_time,
        "fkg_total_time_seconds": train_time + test_time,
        "fkg_accuracy": metrics["accuracy"],
        "fkg_precision": metrics["precision"],
        "fkg_recall": metrics["recall"],
        "fkg_specificity": metrics["specificity"],
        "fkg_f1": metrics["f1"],
        "fkg_auc": metrics["auc"],
        "fkg_positive_label": positive_label,
        "fkg_train_samples": len(train_records),
        "fkg_test_samples": len(test_records),
        "fkg_feature_count": len(train_records[0]) - 1,
        "fkg_results_csv": str(results_path),
        "fkg_predictions_csv": str(predictions_path),
        "fkg_confusion_matrix_csv": str(conf_matrix_csv),
        "fkg_confusion_matrix_png": str(conf_matrix_path),
        "fkg_scores_png": str(scores_path),
    }


def run_fkg_for_record(record, backend):
    fis = record.get("fis") or {}
    train_rule_path = fis.get("train_rule_path")
    test_rule_path = fis.get("test_rule_path")
    output_dir = fis.get("output_dir")
    if not train_rule_path or not test_rule_path or not output_dir:
        raise ValueError(f"Fold {record.get('modality')} #{record.get('fold')} has no FIS rule output.")

    print("__________Running FKG KFold___________")
    print(f"Modality={record['modality']}; fold={record['fold']}/{record['folds']}; backend={backend}")
    fkg_stage_start = time.perf_counter()
    fkg_record = run_native_fkg_for_rules(
        train_rule_path=Path(train_rule_path),
        test_rule_path=Path(test_rule_path),
        output_dir=Path(output_dir),
        modality=f"{record['display_name']}/fold_{int(record['fold']):02d}",
        backend=backend,
    )
    fkg_record["fkg_stage_time_seconds"] = time.perf_counter() - fkg_stage_start
    record["fkg"] = fkg_record

    train_csv = Path(record["train_csv"])
    metadata_path = train_csv.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["fkg"] = fkg_record
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return fkg_record


def run_fkg_for_manifests(manifests, backend):
    for manifest in manifests:
        for record in manifest.get("fold_records", []):
            run_fkg_for_record(record, backend)

        manifest_path = Path(manifest["output_root"]) / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifests


def flatten_fold_records(manifests):
    rows = []
    for manifest in manifests:
        for record in manifest["fold_records"]:
            fis = record.get("fis") or {}
            fkg = record.get("fkg") or {}
            preprocess_time = record.get("preprocess_time_seconds")
            fis_stage_time = fis.get("fis_stage_time_seconds")
            fkg_stage_time = fkg.get("fkg_stage_time_seconds")
            fis_total_time = (
                None
                if fis.get("train_time_seconds") is None and fis.get("fis_eval_time_seconds") is None
                else (fis.get("train_time_seconds") or 0) + (fis.get("fis_eval_time_seconds") or 0)
            )
            fkg_total_time = fkg.get("fkg_total_time_seconds")
            fis_end_to_end_time = (
                None
                if fis_stage_time is None
                else (preprocess_time or 0) + fis_stage_time
            )
            fkg_end_to_end_time = (
                None
                if fkg_stage_time is None
                else (preprocess_time or 0) + (fis_stage_time or fis_total_time or 0) + fkg_stage_time
            )
            fkg_full_train_time = None
            if fkg_end_to_end_time is not None and fkg.get("fkg_test_time_seconds") is not None:
                fkg_full_train_time = fkg_end_to_end_time - fkg.get("fkg_test_time_seconds")
            fold_total_time = record.get("fold_total_time_seconds") or fkg_end_to_end_time or fis_end_to_end_time or preprocess_time
            row = {
                "modality": record["modality"],
                "fold": record["fold"],
                "folds": record["folds"],
                "seed": record["seed"],
                "splitter": record.get("splitter"),
                "split_group_column": record.get("split_group_column"),
                "patient_id_source": record.get("patient_id_source"),
                "feature_selection_fit": record["feature_selection_fit"],
                "fis_range_source": record["fis_range_source"],
                "smote": record["smote"],
                "train_source_rows": record.get("train_source_rows"),
                "test_source_rows": record.get("test_source_rows"),
                "train_rows": record["train_rows"],
                "test_rows": record["test_rows"],
                "train_patient_count": record.get("train_patient_count"),
                "test_patient_count": record.get("test_patient_count"),
                "patient_overlap_count": record.get("patient_overlap_count"),
                "train_ids_csv": record.get("train_ids_csv"),
                "test_ids_csv": record.get("test_ids_csv"),
                "feature_count": record["feature_count"],
                "preprocess_time_seconds": preprocess_time,
                "fis_stage_time_seconds": fis_stage_time,
                "fis_end_to_end_time_seconds": fis_end_to_end_time,
                "fkg_stage_time_seconds": fkg_stage_time,
                "fkg_end_to_end_time_seconds": fkg_end_to_end_time,
                "fkg_full_train_time_seconds": fkg_full_train_time,
                "fold_total_time_seconds": fold_total_time,
                "cluster": json.dumps(record["cluster"]),
                "train_csv": record["train_csv"],
                "test_csv": record["test_csv"],
                "selected_features_csv": record["selected_features_csv"],
                "fis_engine": fis.get("fis_engine"),
                "fis_backend": fis.get("fis_backend"),
                "native_module_dir": fis.get("native_module_dir"),
                "gpu_compiled": fis.get("gpu_compiled"),
                "gpu_available": fis.get("gpu_available"),
                "fis_file_name": fis.get("file_name"),
                "fis_train_time_seconds": fis.get("train_time_seconds"),
                "fis_eval_time_seconds": fis.get("fis_eval_time_seconds"),
                "fis_total_time_seconds": fis_total_time,
                "fis_accuracy": fis.get("fis_accuracy"),
                "fis_precision": fis.get("fis_precision"),
                "fis_recall": fis.get("fis_recall"),
                "fis_f1": fis.get("fis_f1"),
                "fis_output_dir": fis.get("output_dir"),
                "fis_results_csv": fis.get("fis_results_csv"),
                "fis_predictions_csv": fis.get("fis_predictions_csv"),
                "fis_confusion_matrix_png": fis.get("fis_confusion_matrix_png"),
                "fis_scores_png": fis.get("fis_scores_png"),
                "train_rule_path": fis.get("train_rule_path"),
                "test_rule_path": fis.get("test_rule_path"),
                "fkg_engine": fkg.get("fkg_engine"),
                "fkg_backend_request": fkg.get("fkg_backend_request"),
                "fkg_backend": fkg.get("fkg_backend"),
                "fkg_native_module_dir": fkg.get("fkg_native_module_dir"),
                "fkg_gpu_compiled": fkg.get("fkg_gpu_compiled"),
                "fkg_gpu_available": fkg.get("fkg_gpu_available"),
                "fkg_train_time_seconds": fkg.get("fkg_train_time_seconds"),
                "fkg_test_time_seconds": fkg.get("fkg_test_time_seconds"),
                "fkg_total_time_seconds": fkg_total_time,
                "fkg_accuracy": fkg.get("fkg_accuracy"),
                "fkg_precision": fkg.get("fkg_precision"),
                "fkg_recall": fkg.get("fkg_recall"),
                "fkg_specificity": fkg.get("fkg_specificity"),
                "fkg_f1": fkg.get("fkg_f1"),
                "fkg_auc": fkg.get("fkg_auc"),
                "fkg_positive_label": fkg.get("fkg_positive_label"),
                "fkg_train_samples": fkg.get("fkg_train_samples"),
                "fkg_test_samples": fkg.get("fkg_test_samples"),
                "fkg_feature_count": fkg.get("fkg_feature_count"),
                "fkg_results_csv": fkg.get("fkg_results_csv"),
                "fkg_predictions_csv": fkg.get("fkg_predictions_csv"),
                "fkg_confusion_matrix_csv": fkg.get("fkg_confusion_matrix_csv"),
                "fkg_confusion_matrix_png": fkg.get("fkg_confusion_matrix_png"),
                "fkg_scores_png": fkg.get("fkg_scores_png"),
                "fkgs_count": len(record.get("fkgs") or []),
            }
            rows.append(row)
    return rows


def flatten_fkgs_records(manifests):
    rows = []
    for manifest in manifests:
        for record in manifest.get("fold_records", []):
            fis = record.get("fis") or {}
            preprocess_time = record.get("preprocess_time_seconds") or 0
            fis_stage_time = fis.get("fis_stage_time_seconds") or 0

            for fkgs in record.get("fkgs") or []:
                fkgs_stage_time = fkgs.get("fkgs_stage_time_seconds")
                fkgs_algorithm_time = (
                    (fkgs.get("sampling_time_mean") or 0)
                    + (fkgs.get("total_time_mean") or 0)
                )
                fkgs_end_to_end_time = None
                if fkgs.get("total_time_mean") is not None:
                    fkgs_end_to_end_time = preprocess_time + fis_stage_time + fkgs_algorithm_time
                fkgs_full_train_time = None
                if fkgs_end_to_end_time is not None and fkgs.get("test_time_mean") is not None:
                    fkgs_full_train_time = fkgs_end_to_end_time - fkgs.get("test_time_mean")
                rows.append(
                    {
                        "modality": record["modality"],
                        "fold": record["fold"],
                        "folds": record["folds"],
                        "seed": fkgs.get("seed", record["seed"]),
                        "splitter": record.get("splitter"),
                        "split_group_column": record.get("split_group_column"),
                        "patient_id_source": record.get("patient_id_source"),
                        "ran": fkgs.get("ran"),
                        "epsilon": fkgs.get("e"),
                        "turns": fkgs.get("turns"),
                        "train_source_rows": record.get("train_source_rows"),
                        "test_source_rows": record.get("test_source_rows"),
                        "train_rows": record["train_rows"],
                        "test_rows": record["test_rows"],
                        "train_patient_count": record.get("train_patient_count"),
                        "test_patient_count": record.get("test_patient_count"),
                        "patient_overlap_count": record.get("patient_overlap_count"),
                        "feature_count": record["feature_count"],
                        "preprocess_time_seconds": record.get("preprocess_time_seconds"),
                        "fis_stage_time_seconds": fis.get("fis_stage_time_seconds"),
                        "fkgs_stage_time_seconds": fkgs_stage_time,
                        "fkgs_end_to_end_time_seconds": fkgs_end_to_end_time,
                        "fkgs_sampling_time_seconds": fkgs.get("sampling_time_mean"),
                        "fkgs_sampling_time_std_seconds": fkgs.get("sampling_time_std"),
                        "fkgs_train_time_seconds": fkgs.get("train_time_mean"),
                        "fkgs_train_time_std_seconds": fkgs.get("train_time_std"),
                        "fkgs_full_train_time_seconds": fkgs_full_train_time,
                        "fkgs_test_time_seconds": fkgs.get("test_time_mean"),
                        "fkgs_test_time_std_seconds": fkgs.get("test_time_std"),
                        "fkgs_total_time_seconds": fkgs.get("total_time_mean"),
                        "fkgs_total_time_std_seconds": fkgs.get("total_time_std"),
                        "fkgs_accuracy_pct": fkgs.get("accuracy_mean"),
                        "fkgs_accuracy_std_pct": fkgs.get("accuracy_std"),
                        "fkgs_precision_pct": fkgs.get("precision_mean"),
                        "fkgs_precision_std_pct": fkgs.get("precision_std"),
                        "fkgs_recall_pct": fkgs.get("recall_mean"),
                        "fkgs_recall_std_pct": fkgs.get("recall_std"),
                        "fkgs_module_path": fkgs.get("fkgs_module_path"),
                        "fkgs_bar_scores_png": fkgs.get("bar_scores_png"),
                        "fis_file_name": fis.get("file_name"),
                        "train_rule_path": fis.get("train_rule_path"),
                        "test_rule_path": fis.get("test_rule_path"),
                    }
                )
    return rows


def save_summary_bar_chart(summary_df, value_column, output_path, title, ylabel):
    plot_df = summary_df.dropna(subset=[value_column])
    if plot_df.empty:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"{row.modality}\nfold {int(row.fold)}" for row in plot_df.itertuples()]
        values = plot_df[value_column].astype(float).tolist()
        colors = ["#4C78A8" if modality == "image" else "#F58518" if modality == "table" else "#54A24B" for modality in plot_df["modality"]]

        plt.figure(figsize=(max(9, len(labels) * 0.65), 5.5))
        bars = plt.bar(labels, values, color=colors)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=30, ha="right")
        for bar, value in zip(bars, values):
            text = f"{value:.2f}" if value > 1 else f"{value:.2%}"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Summary chart skipped for {value_column}: {exc}")


def build_modality_mean_std_summary(summary_df, numeric_columns):
    if summary_df.empty or not numeric_columns:
        return pd.DataFrame()

    numeric_df = summary_df.copy()
    for column in numeric_columns:
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

    grouped = numeric_df.groupby("modality", dropna=False)
    parts = [grouped.size().rename("folds")]
    ordered_columns = ["modality", "folds"]

    for column in numeric_columns:
        mean_column = f"{column}_mean"
        std_column = f"{column}_std"
        parts.append(grouped[column].mean().rename(mean_column))
        parts.append(grouped[column].std(ddof=1).rename(std_column))
        ordered_columns.extend([mean_column, std_column])

    stats_df = pd.concat(parts, axis=1).reset_index()
    return stats_df[[column for column in ordered_columns if column in stats_df.columns]]


def build_fkgs_mean_std_summary(fkgs_df):
    if fkgs_df.empty:
        return pd.DataFrame()

    numeric_columns = [
        "train_source_rows",
        "test_source_rows",
        "train_rows",
        "test_rows",
        "train_patient_count",
        "test_patient_count",
        "patient_overlap_count",
        "preprocess_time_seconds",
        "fis_stage_time_seconds",
        "fkgs_stage_time_seconds",
        "fkgs_end_to_end_time_seconds",
        "feature_count",
        "fkgs_sampling_time_seconds",
        "fkgs_train_time_seconds",
        "fkgs_full_train_time_seconds",
        "fkgs_test_time_seconds",
        "fkgs_total_time_seconds",
        "fkgs_accuracy_pct",
        "fkgs_precision_pct",
        "fkgs_recall_pct",
    ]
    numeric_df = fkgs_df.copy()
    for column in numeric_columns:
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

    group_columns = ["modality", "ran", "epsilon"]
    grouped = numeric_df.groupby(group_columns, dropna=False)
    parts = [grouped.size().rename("folds")]
    ordered_columns = group_columns + ["folds"]
    for column in numeric_columns:
        mean_column = f"{column}_mean"
        std_column = f"{column}_std"
        parts.append(grouped[column].mean().rename(mean_column))
        parts.append(grouped[column].std(ddof=1).rename(std_column))
        ordered_columns.extend([mean_column, std_column])

    stats_df = pd.concat(parts, axis=1).reset_index()
    return stats_df[[column for column in ordered_columns if column in stats_df.columns]]


def write_fkgs_tables(stats_df, csv_path, markdown_path):
    table_configs = [(15, 0.2), (15, 0.3), (20, 0.2), (20, 0.3)]
    modality_order = [
        "table",
        "image",
        "fusion",
        "fusion_filter",
        "fusion_hadamard",
        "fusion_tensor",
        "fusion_wrapper",
    ]
    model_by_modality = {
        "table": "FKG-UM",
        "image": "FKG-UM",
        "fusion": "FKG-MM",
        "fusion_filter": "FKG-MM",
        "fusion_hadamard": "FKG-MM",
        "fusion_tensor": "FKG-MM",
        "fusion_wrapper": "FKG-MM",
    }
    label_by_modality = {
        "table": "Du lieu dang bang full",
        "image": "Du lieu anh",
        "fusion": "Du lieu anh+bang",
        "fusion_filter": "Fusion Filter",
        "fusion_hadamard": "Fusion Hadamard",
        "fusion_tensor": "Fusion Tensor",
        "fusion_wrapper": "Fusion Wrapper",
    }
    title_by_config = {
        (15, 0.2): "Bang 3.2. Phuong phap lua chon thuoc tinh voi ti le mau 15% va nguong sai so 0.2",
        (15, 0.3): "Bang 3.3. Phuong phap lua chon thuoc tinh voi ti le mau 15% va nguong sai so 0.3",
        (20, 0.2): "Bang 3.4. Phuong phap lua chon thuoc tinh voi ti le mau 20% va nguong sai so 0.2",
        (20, 0.3): "Bang 3.5. Phuong phap lua chon thuoc tinh voi ti le mau 20% va nguong sai so 0.3",
    }
    table_label_by_config = {
        (15, 0.2): "Bang 3.2",
        (15, 0.3): "Bang 3.3",
        (20, 0.2): "Bang 3.4",
        (20, 0.3): "Bang 3.5",
    }

    table_rows = []
    if not stats_df.empty:
        for ran, epsilon in table_configs:
            subset = stats_df[
                (pd.to_numeric(stats_df["ran"], errors="coerce") == ran)
                & (pd.to_numeric(stats_df["epsilon"], errors="coerce").sub(epsilon).abs() < 1e-9)
            ].copy()
            for modality in modality_order:
                row_df = subset[subset["modality"] == modality]
                if row_df.empty:
                    continue
                row = row_df.iloc[0]
                table_rows.append(
                    {
                        "table": table_label_by_config[(ran, epsilon)],
                        "ran": ran,
                        "epsilon": epsilon,
                        "model": model_by_modality[modality],
                        "modality": label_by_modality[modality],
                        "modality_key": modality,
                        "folds": row.get("folds"),
                        "selected_feature_count": row.get("feature_count_mean"),
                        "accuracy_pct": row.get("fkgs_accuracy_pct_mean"),
                        "accuracy_std_pct": row.get("fkgs_accuracy_pct_std"),
                        "train_time_s": row.get("fkgs_full_train_time_seconds_mean"),
                        "train_time_std_s": row.get("fkgs_full_train_time_seconds_std"),
                        "test_time_s": row.get("fkgs_test_time_seconds_mean"),
                        "test_time_std_s": row.get("fkgs_test_time_seconds_std"),
                        "algorithm_total_time_s": row.get("fkgs_total_time_seconds_mean"),
                        "algorithm_total_time_std_s": row.get("fkgs_total_time_seconds_std"),
                        "end_to_end_time_s": row.get("fkgs_end_to_end_time_seconds_mean"),
                        "end_to_end_time_std_s": row.get("fkgs_end_to_end_time_seconds_std"),
                    }
                )

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(csv_path, index=False)

    lines = [
        "# KFold FKGS summary tables",
        "",
        "Note: KFold uses 5 folds. Timing starts from precomputed feature CSVs, so it does not include raw image preprocessing/feature extraction. Training time is fold preprocessing over those CSVs + FIS + FKGS sampling/train. Test time is FKGS test time. Total time is training + test.",
        "Selected feature counts are stored in selected_feature_count.",
        "",
    ]
    for config in table_configs:
        title = title_by_config[config]
        lines.extend(
            [
                f"## {title}",
                "",
                "| Mo hinh | Mo thuc | Acc (%) | Thoi gian huan luyen | Thoi gian kiem tra | Tong thoi gian (s) |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        subset = table_df[
            (pd.to_numeric(table_df.get("ran", pd.Series(dtype=float)), errors="coerce") == config[0])
            & (pd.to_numeric(table_df.get("epsilon", pd.Series(dtype=float)), errors="coerce").sub(config[1]).abs() < 1e-9)
        ]
        if subset.empty:
            lines.append("|  | Chua co ket qua |  |  |  |  |")
        else:
            subset = subset.sort_values(
                by="modality_key",
                key=lambda series: series.map({name: idx for idx, name in enumerate(modality_order)}),
            )
            for row in subset.to_dict("records"):
                lines.append(
                    "| {model} | {modality} | {acc:.2f} | {train:.2f} | {test:.2f} | {total:.2f} |".format(
                        model=row["model"],
                        modality=row["modality"],
                        acc=float(row["accuracy_pct"]),
                        train=float(row["train_time_s"]),
                        test=float(row["test_time_s"]),
                        total=float(row["end_to_end_time_s"]),
                    )
                )
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def write_fkgs_report_outputs(manifests, report_root):
    rows = flatten_fkgs_records(manifests)
    fkgs_df = pd.DataFrame(rows)
    fkgs_csv = report_root / "kfold_fkgs_run_summary.csv"
    fkgs_mean_std_csv = report_root / "kfold_fkgs_mean_std_summary.csv"
    fkgs_tables_csv = report_root / "kfold_fkgs_tables.csv"
    fkgs_tables_md = report_root / "kfold_fkgs_tables.md"

    fkgs_df.to_csv(fkgs_csv, index=False)
    stats_df = build_fkgs_mean_std_summary(fkgs_df)
    stats_df.to_csv(fkgs_mean_std_csv, index=False)
    write_fkgs_tables(stats_df, fkgs_tables_csv, fkgs_tables_md)

    return {
        "fkgs_summary_csv": str(fkgs_csv),
        "fkgs_mean_std_csv": str(fkgs_mean_std_csv),
        "fkgs_tables_csv": str(fkgs_tables_csv),
        "fkgs_tables_md": str(fkgs_tables_md),
    }


def save_modality_errorbar_chart(stats_df, value_column, output_path, title, ylabel, percent=False):
    mean_column = f"{value_column}_mean"
    std_column = f"{value_column}_std"
    if stats_df.empty or mean_column not in stats_df.columns:
        return

    plot_df = stats_df.dropna(subset=[mean_column])
    if plot_df.empty:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = plot_df["modality"].astype(str).tolist()
        values = plot_df[mean_column].astype(float).to_numpy()
        errors = (
            plot_df[std_column].fillna(0).astype(float).to_numpy()
            if std_column in plot_df.columns
            else np.zeros(len(values))
        )
        colors = ["#4C78A8" if modality == "image" else "#F58518" if modality == "table" else "#54A24B" for modality in plot_df["modality"]]

        plt.figure(figsize=(max(7, len(labels) * 1.8), 5.5))
        bars = plt.bar(labels, values, yerr=errors, capsize=6, color=colors, ecolor="#333333")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=0)

        upper = float(np.nanmax(values + errors)) if len(values) else 0.0
        y_offset = max(upper * 0.03, 0.02 if percent else 0.05)
        for bar, value, error in zip(bars, values, errors):
            if percent:
                text = f"{value:.2%}\nstd {error:.2%}"
            else:
                text = f"{value:.2f}\nstd {error:.2f}"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + y_offset,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.ylim(top=upper + y_offset * 4)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Mean/std chart skipped for {value_column}: {exc}")


def save_fkg_mean_std_table(stats_df, output_path):
    if stats_df.empty:
        return

    required_columns = [
        "modality",
        "fkg_total_time_seconds_mean",
        "fkg_total_time_seconds_std",
        "fkg_end_to_end_time_seconds_mean",
        "fkg_end_to_end_time_seconds_std",
        "fkg_accuracy_mean",
        "fkg_accuracy_std",
        "fkg_precision_mean",
        "fkg_precision_std",
        "fkg_recall_mean",
        "fkg_recall_std",
        "fkg_specificity_mean",
        "fkg_specificity_std",
        "fkg_f1_mean",
        "fkg_f1_std",
        "fkg_auc_mean",
        "fkg_auc_std",
    ]
    if not all(column in stats_df.columns for column in required_columns):
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        view = stats_df[required_columns].copy()
        view.columns = [
            "Data",
            "FKG algo mean (s)",
            "FKG algo std (s)",
            "End-to-end mean (s)",
            "End-to-end std (s)",
            "Acc mean",
            "Acc std",
            "Precision mean",
            "Precision std",
            "Recall mean",
            "Recall std",
            "Specificity mean",
            "Specificity std",
            "F1 mean",
            "F1 std",
            "AUC mean",
            "AUC std",
        ]
        for column in [
            "Acc mean",
            "Acc std",
            "Precision mean",
            "Precision std",
            "Recall mean",
            "Recall std",
            "Specificity mean",
            "Specificity std",
            "F1 mean",
            "F1 std",
            "AUC mean",
            "AUC std",
        ]:
            view[column] = pd.to_numeric(view[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value * 100:.2f}%"
            )
        for column in [
            "FKG algo mean (s)",
            "FKG algo std (s)",
            "End-to-end mean (s)",
            "End-to-end std (s)",
        ]:
            view[column] = pd.to_numeric(view[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )

        fig, ax = plt.subplots(figsize=(22, 3.2))
        ax.axis("off")
        table = ax.table(
            cellText=view.values,
            colLabels=view.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.65)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D7DE")
            if row == 0:
                cell.set_facecolor("#1F2937")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F6F8FA")
            else:
                cell.set_facecolor("white")
        plt.title("FKG KFold Mean +/- Std Summary", fontsize=14, weight="bold", pad=12)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] FKG mean/std table skipped: {exc}")


def save_fkg_accuracy_mean_std_table(stats_df, output_path):
    required_columns = [
        "modality",
        "fkg_accuracy_mean",
        "fkg_accuracy_std",
    ]
    if stats_df.empty or not all(column in stats_df.columns for column in required_columns):
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        view = stats_df[required_columns].copy()
        view["accuracy_mean_percent"] = pd.to_numeric(view["fkg_accuracy_mean"], errors="coerce") * 100
        view["accuracy_std_percent"] = pd.to_numeric(view["fkg_accuracy_std"], errors="coerce") * 100
        view["accuracy_mean_std"] = view.apply(
            lambda row: (
                ""
                if pd.isna(row["accuracy_mean_percent"]) or pd.isna(row["accuracy_std_percent"])
                else f'{row["accuracy_mean_percent"]:.2f}% +/- {row["accuracy_std_percent"]:.2f}%'
            ),
            axis=1,
        )
        table_df = pd.DataFrame(
            {
                "Data": view["modality"].astype(str),
                "Accuracy mean": view["accuracy_mean_percent"].map(
                    lambda value: "" if pd.isna(value) else f"{value:.2f}%"
                ),
                "Stdev": view["accuracy_std_percent"].map(
                    lambda value: "" if pd.isna(value) else f"{value:.2f}%"
                ),
                "Acc +/- stdev": view["accuracy_mean_std"],
            }
        )

        fig, ax = plt.subplots(figsize=(9.5, 2.8))
        ax.axis("off")
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D7DE")
            if row == 0:
                cell.set_facecolor("#1F2937")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F6F8FA")
            else:
                cell.set_facecolor("white")
        plt.title("FKG KFold Accuracy Mean +/- Stdev", fontsize=15, weight="bold", pad=14)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] FKG accuracy mean/std table skipped: {exc}")


def write_report_outputs(manifests, report_root):
    report_root.mkdir(parents=True, exist_ok=True)
    rows = flatten_fold_records(manifests)
    summary_df = pd.DataFrame(rows)
    summary_csv = report_root / "kfold_run_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    modality_average_csv = report_root / "kfold_modality_average_summary.csv"
    modality_mean_std_csv = report_root / "kfold_modality_mean_std_summary.csv"
    stats_df = pd.DataFrame()
    if not summary_df.empty:
        average_columns = [
            "train_source_rows",
            "test_source_rows",
            "train_rows",
            "test_rows",
            "train_patient_count",
            "test_patient_count",
            "patient_overlap_count",
            "feature_count",
            "preprocess_time_seconds",
            "fis_stage_time_seconds",
            "fis_end_to_end_time_seconds",
            "fis_train_time_seconds",
            "fis_eval_time_seconds",
            "fis_total_time_seconds",
            "fis_accuracy",
            "fis_precision",
            "fis_recall",
            "fis_f1",
            "fkg_stage_time_seconds",
            "fkg_end_to_end_time_seconds",
            "fkg_full_train_time_seconds",
            "fkg_train_time_seconds",
            "fkg_test_time_seconds",
            "fkg_total_time_seconds",
            "fkg_accuracy",
            "fkg_precision",
            "fkg_recall",
            "fkg_specificity",
            "fkg_f1",
            "fkg_auc",
            "fold_total_time_seconds",
        ]
        average_columns = [
            column
            for column in average_columns
            if column in summary_df.columns
        ]
        for column in average_columns:
            summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")
        average_df = (
            summary_df.groupby("modality", dropna=False)[average_columns]
            .mean(numeric_only=True)
            .reset_index()
        )
        average_df.insert(1, "folds", summary_df.groupby("modality", dropna=False).size().values)
        average_df.to_csv(modality_average_csv, index=False)
        stats_df = build_modality_mean_std_summary(summary_df, average_columns)
        stats_df.to_csv(modality_mean_std_csv, index=False)
    else:
        pd.DataFrame().to_csv(modality_average_csv, index=False)
        pd.DataFrame().to_csv(modality_mean_std_csv, index=False)

    modality_summary_paths = {}
    for modality, modality_df in summary_df.groupby("modality", dropna=False):
        modality_dir = report_root / str(modality)
        modality_dir.mkdir(parents=True, exist_ok=True)
        modality_path = modality_dir / "fold_summary.csv"
        modality_df.to_csv(modality_path, index=False)
        modality_summary_paths[str(modality)] = str(modality_path)

    save_summary_bar_chart(
        summary_df,
        "fis_total_time_seconds",
        report_root / "fis_total_time_by_fold.png",
        "FIS Total Time By Fold",
        "Seconds",
    )
    save_summary_bar_chart(
        summary_df,
        "fis_accuracy",
        report_root / "fis_accuracy_by_fold.png",
        "FIS Accuracy By Fold",
        "Accuracy",
    )
    save_summary_bar_chart(
        summary_df,
        "feature_count",
        report_root / "selected_feature_count_by_fold.png",
        "Selected Feature Count By Fold",
        "Feature count",
    )
    save_summary_bar_chart(
        summary_df,
        "fkg_total_time_seconds",
        report_root / "fkg_total_time_by_fold.png",
        "FKG Total Time By Fold",
        "Seconds",
    )
    save_summary_bar_chart(
        summary_df,
        "fkg_accuracy",
        report_root / "fkg_accuracy_by_fold.png",
        "FKG Accuracy By Fold",
        "Accuracy",
    )
    save_summary_bar_chart(
        summary_df,
        "preprocess_time_seconds",
        report_root / "preprocess_time_by_fold.png",
        "Preprocess Time By Fold",
        "Seconds",
    )
    save_summary_bar_chart(
        summary_df,
        "fkg_end_to_end_time_seconds",
        report_root / "fkg_end_to_end_time_by_fold.png",
        "FKG End-to-End Time By Fold",
        "Seconds",
    )
    save_summary_bar_chart(
        summary_df,
        "fold_total_time_seconds",
        report_root / "fold_total_time_by_fold.png",
        "Fold Total Time By Fold",
        "Seconds",
    )
    save_modality_errorbar_chart(
        stats_df,
        "fis_total_time_seconds",
        report_root / "fis_total_time_mean_std.png",
        "FIS Total Time Mean +/- Std",
        "Seconds",
    )
    save_modality_errorbar_chart(
        stats_df,
        "fis_accuracy",
        report_root / "fis_accuracy_mean_std.png",
        "FIS Accuracy Mean +/- Std",
        "Accuracy",
        percent=True,
    )
    save_modality_errorbar_chart(
        stats_df,
        "fkg_total_time_seconds",
        report_root / "fkg_total_time_mean_std.png",
        "FKG Total Time Mean +/- Std",
        "Seconds",
    )
    save_modality_errorbar_chart(
        stats_df,
        "fkg_accuracy",
        report_root / "fkg_accuracy_mean_std.png",
        "FKG Accuracy Mean +/- Std",
        "Accuracy",
        percent=True,
    )
    save_modality_errorbar_chart(
        stats_df,
        "preprocess_time_seconds",
        report_root / "preprocess_time_mean_std.png",
        "Preprocess Time Mean +/- Std",
        "Seconds",
    )
    save_modality_errorbar_chart(
        stats_df,
        "fkg_end_to_end_time_seconds",
        report_root / "fkg_end_to_end_time_mean_std.png",
        "FKG End-to-End Time Mean +/- Std",
        "Seconds",
    )
    save_modality_errorbar_chart(
        stats_df,
        "fold_total_time_seconds",
        report_root / "fold_total_time_mean_std.png",
        "Fold Total Time Mean +/- Std",
        "Seconds",
    )
    save_fkg_mean_std_table(stats_df, report_root / "fkg_mean_std_table.png")
    save_fkg_accuracy_mean_std_table(stats_df, report_root / "fkg_accuracy_mean_std_table.png")
    fkgs_outputs = write_fkgs_report_outputs(manifests, report_root)

    outputs = {
        "summary_csv": str(summary_csv),
        "modality_average_csv": str(modality_average_csv),
        "modality_mean_std_csv": str(modality_mean_std_csv),
        "modality_summary_csv": modality_summary_paths,
        "fis_total_time_png": str(report_root / "fis_total_time_by_fold.png"),
        "fis_accuracy_png": str(report_root / "fis_accuracy_by_fold.png"),
        "fkg_total_time_png": str(report_root / "fkg_total_time_by_fold.png"),
        "fkg_accuracy_png": str(report_root / "fkg_accuracy_by_fold.png"),
        "preprocess_time_png": str(report_root / "preprocess_time_by_fold.png"),
        "fkg_end_to_end_time_png": str(report_root / "fkg_end_to_end_time_by_fold.png"),
        "fold_total_time_png": str(report_root / "fold_total_time_by_fold.png"),
        "fis_total_time_mean_std_png": str(report_root / "fis_total_time_mean_std.png"),
        "fis_accuracy_mean_std_png": str(report_root / "fis_accuracy_mean_std.png"),
        "fkg_total_time_mean_std_png": str(report_root / "fkg_total_time_mean_std.png"),
        "fkg_accuracy_mean_std_png": str(report_root / "fkg_accuracy_mean_std.png"),
        "preprocess_time_mean_std_png": str(report_root / "preprocess_time_mean_std.png"),
        "fkg_end_to_end_time_mean_std_png": str(report_root / "fkg_end_to_end_time_mean_std.png"),
        "fold_total_time_mean_std_png": str(report_root / "fold_total_time_mean_std.png"),
        "fkg_mean_std_table_png": str(report_root / "fkg_mean_std_table.png"),
        "fkg_accuracy_mean_std_table_png": str(report_root / "fkg_accuracy_mean_std_table.png"),
        "selected_feature_count_png": str(report_root / "selected_feature_count_by_fold.png"),
    }
    outputs.update(fkgs_outputs)
    return outputs


def import_fkgs_legacy_class():
    import site

    for module_name in ["module.FKG.FKG_S", "fisa_module"]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    candidate_site_paths = []
    try:
        candidate_site_paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidate_site_paths.append(site.getusersitepackages())
    except Exception:
        pass

    for site_path in reversed([path for path in candidate_site_paths if path]):
        if site_path in sys.path:
            sys.path.remove(site_path)
        sys.path.insert(0, site_path)

    from module.FKG.FKG_S import FKGS

    legacy_module = sys.modules.get("fisa_module")
    if legacy_module is None or not hasattr(legacy_module, "calculateA"):
        module_path = "" if legacy_module is None else str(getattr(legacy_module, "__file__", ""))
        raise ImportError(f"FKGS requires legacy fisa_module.calculateA, got: {module_path}")
    return FKGS, str(getattr(legacy_module, "__file__", ""))


def run_fkgs_for_rules(
    train_rule_path,
    test_rule_path,
    modality,
    ran_values,
    e_values,
    seed_base,
    turns,
    reuse_existing,
    workers,
):
    summaries = []
    pending = []
    helper_script = CURRENT_DIR / "run_fkgs_once.py"
    fkgs_output_dir = PROJECT_ROOT / "data" / "FKG" / Path(modality)

    for ran in ran_values:
        for e_value in e_values:
            seed = int(seed_base + ran * 1000 + round(float(e_value) * 1000))
            summary_path = fkgs_output_dir / f"fkgs_summary_ran{ran}_e{e_value}_seed{seed}.json"
            print("__________Running FKG-S___________")
            print(f"Modality={modality}; ran={ran}; e={e_value}")
            if reuse_existing and summary_path.exists():
                print(f"[FKGS-REUSE] {summary_path}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["fkgs_stage_time_seconds"] = 0.0
                summary["summary_json"] = str(summary_path)
                summaries.append(summary)
                print("-" * 100)
                continue

            pending.append(
                {
                    "ran": ran,
                    "epsilon": e_value,
                    "seed": seed,
                    "summary_path": summary_path,
                    "log_path": fkgs_output_dir / f"fkgs_ran{ran}_e{e_value}_seed{seed}.log",
                }
            )

    if not pending:
        return summaries

    workers = max(1, int(workers or 1))
    running = []
    pending_index = 0

    def start_job(job):
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("MPLBACKEND", "Agg")
        job["summary_path"].parent.mkdir(parents=True, exist_ok=True)
        log_file = open(job["log_path"], "w", encoding="utf-8")
        command = [
            sys.executable,
            "-u",
            str(helper_script),
            "--train-rule",
            str(train_rule_path),
            "--test-rule",
            str(test_rule_path),
            "--modality",
            modality,
            "--ran",
            str(job["ran"]),
            "--epsilon",
            str(job["epsilon"]),
            "--turns",
            str(turns),
            "--seed",
            str(job["seed"]),
            "--summary-path",
            str(job["summary_path"]),
        ]
        print(f"[FKGS-START] ran={job['ran']} e={job['epsilon']} log={job['log_path']}")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT.parent),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        job["process"] = process
        job["log_file"] = log_file
        job["started_at"] = time.perf_counter()
        running.append(job)

    def finish_job(job):
        job["log_file"].close()
        elapsed = time.perf_counter() - job["started_at"]
        if job["process"].returncode != 0:
            tail = ""
            if job["log_path"].exists():
                tail = "\n".join(job["log_path"].read_text(encoding="utf-8", errors="ignore").splitlines()[-40:])
            raise subprocess.CalledProcessError(
                job["process"].returncode,
                f"FKGS ran={job['ran']} e={job['epsilon']}",
                output=tail,
            )
        summary = json.loads(job["summary_path"].read_text(encoding="utf-8"))
        summary["fkgs_stage_time_seconds"] = elapsed
        summary["summary_json"] = str(job["summary_path"])
        summary["log_path"] = str(job["log_path"])
        summaries.append(summary)
        print(f"[FKGS-DONE] ran={job['ran']} e={job['epsilon']} elapsed={elapsed:.2f}s")
        print("-" * 100)

    while pending_index < len(pending) or running:
        while pending_index < len(pending) and len(running) < workers:
            start_job(pending[pending_index])
            pending_index += 1

        time.sleep(5)
        for job in running[:]:
            if job["process"].poll() is None:
                continue
            running.remove(job)
            finish_job(job)

    return summaries


def prepare_modality(config, args):
    source_path = resolve_project_path(getattr(args, config.source_arg))
    output_root = resolve_project_path(args.output_root) / config.key
    output_root.mkdir(parents=True, exist_ok=True)

    features, labels, ids, label_mapping, patient_id_source_used = load_source_frame(
        source_path,
        args.patient_id_source,
    )
    features, labels, ids = limit_rows(features, labels, ids, args.max_rows, args.seed)

    splitter_name, patient_groups, splits = build_patient_group_splits(
        features,
        labels,
        ids,
        args.folds,
        args.seed,
    )
    fold_records = []

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        fold_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        fold_seed = args.seed + fold_number
        train_x = features.iloc[train_index].reset_index(drop=True)
        test_x = features.iloc[test_index].reset_index(drop=True)
        train_y = labels.iloc[train_index].reset_index(drop=True)
        test_y = labels.iloc[test_index].reset_index(drop=True)
        train_ids = ids.iloc[train_index].reset_index(drop=True)
        test_ids = ids.iloc[test_index].reset_index(drop=True)
        train_patient_ids = set(patient_groups.iloc[train_index].tolist())
        test_patient_ids = set(patient_groups.iloc[test_index].tolist())
        patient_overlap = train_patient_ids & test_patient_ids

        if config.key == "image":
            train_selected, test_selected, cluster, feature_rows = prepare_image_fold(
                train_x,
                test_x,
                train_y,
                args,
                fold_seed,
            )
        elif config.key == "table":
            train_selected, test_selected, cluster, feature_rows = prepare_table_fold(
                train_x,
                test_x,
                train_y,
                args,
                fold_seed,
            )
        elif config.key == "fusion":
            train_selected, test_selected, cluster, feature_rows = prepare_fusion_fold(
                train_x,
                test_x,
                train_y,
                args,
                fold_seed,
            )
        elif config.key.startswith("fusion_"):
            train_selected, test_selected, cluster, feature_rows = prepare_fusion_variant_fold(
                config.key,
                train_x,
                test_x,
                train_y,
                args,
                fold_seed,
            )
        else:
            raise ValueError(f"Unsupported modality: {config.key}")

        train_selected, test_selected, feature_rows, dropped_constant_columns = drop_train_constant_columns(
            train_selected,
            test_selected,
            feature_rows,
        )
        train_fold = build_fis_frame(train_selected, train_y)
        test_fold = build_fis_frame(test_selected, test_y)
        if dropped_constant_columns:
            cluster = [
                value
                for idx, value in enumerate(cluster[:-1])
                if idx not in set(dropped_constant_columns)
            ] + [cluster[-1]]

        smote_status = "disabled"
        if not args.skip_smote:
            train_fold, smote_status = apply_train_smote(
                train_fold,
                seed=fold_seed,
                max_k_neighbors=args.smote_k_neighbors,
            )

        fold_name = f"fold_{fold_number:02d}"
        fold_dir = output_root / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = fold_dir / "train_data.csv"
        test_path = fold_dir / "test_data.csv"
        train_ids_path = fold_dir / "train_ids.csv"
        test_ids_path = fold_dir / "test_ids.csv"
        selected_features_path = fold_dir / "selected_features.csv"
        metadata_path = fold_dir / "metadata.json"

        train_fold.to_csv(train_path, index=False)
        test_fold.to_csv(test_path, index=False)
        train_ids.to_csv(train_ids_path, index=False)
        test_ids.to_csv(test_ids_path, index=False)
        pd.DataFrame(feature_rows).to_csv(selected_features_path, index=False)
        preprocess_time = time.perf_counter() - preprocess_start

        fis_record = None
        fkg_record = None
        fkgs_records = []
        fis_file_name = f"{config.display_name}/{fold_name}"
        if not args.skip_fis:
            print("__________Running FIS KFold___________")
            print(f"Modality={config.key}; fold={fold_number}/{args.folds}; fileName={fis_file_name}")
            fis_stage_start = time.perf_counter()
            fis_record = run_fis_for_split(
                file_name=fis_file_name,
                train_df=train_fold,
                test_df=test_fold,
                cluster=cluster,
                modality=config.display_name,
                range_source=args.fis_range_source,
                write_fold_heatmap=not args.skip_heatmap,
                run_fis_test=not args.skip_fis_test,
                fis_engine=args.fis_engine,
                native_backend=args.native_backend,
            )
            fis_record["fis_stage_time_seconds"] = time.perf_counter() - fis_stage_start

            if args.run_fkgs:
                fkgs_records = run_fkgs_for_rules(
                    train_rule_path=fis_record["train_rule_path"],
                    test_rule_path=fis_record["test_rule_path"],
                    modality=fis_file_name,
                    ran_values=args.ran,
                    e_values=args.e,
                    seed_base=args.seed + fold_number * 100000,
                    turns=args.fkgs_turns,
                    reuse_existing=args.reuse_fkgs,
                    workers=args.fkgs_workers,
                )

            if args.run_fkg:
                temp_record = {
                    "modality": config.key,
                    "display_name": config.display_name,
                    "fold": fold_number,
                    "folds": args.folds,
                    "train_csv": str(train_path),
                    "fis": fis_record,
                }
                fkg_record = run_fkg_for_record(temp_record, args.fkg_backend)

        fold_total_time = time.perf_counter() - fold_start
        fold_metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "modality": config.key,
            "display_name": config.display_name,
            "source_csv": str(source_path),
            "fold": fold_number,
            "folds": args.folds,
            "seed": args.seed,
            "splitter": splitter_name,
            "split_group_column": PATIENT_ID_COLUMN,
            "patient_id_source": patient_id_source_used,
            "feature_selection_fit": "train_fold_only",
            "fis_range_source": args.fis_range_source,
            "smote": smote_status,
            "label_mapping_encoded_to_original": label_mapping,
            "cluster": cluster,
            "dropped_constant_columns": dropped_constant_columns,
            "train_source_rows": int(len(train_index)),
            "test_source_rows": int(len(test_index)),
            "train_rows": int(train_fold.shape[0]),
            "test_rows": int(test_fold.shape[0]),
            "train_patient_count": int(len(train_patient_ids)),
            "test_patient_count": int(len(test_patient_ids)),
            "patient_overlap_count": int(len(patient_overlap)),
            "patient_overlap_examples": sorted(patient_overlap)[:5],
            "feature_count": int(train_fold.shape[1] - 1),
            "preprocess_time_seconds": preprocess_time,
            "fold_total_time_seconds": fold_total_time,
            "train_csv": str(train_path),
            "test_csv": str(test_path),
            "train_ids_csv": str(train_ids_path),
            "test_ids_csv": str(test_ids_path),
            "selected_features_csv": str(selected_features_path),
            "fis": fis_record,
            "fkg": fkg_record,
            "fkgs": fkgs_records,
        }
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(fold_metadata, file, indent=2)

        fold_records.append(fold_metadata)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modality": config.key,
        "display_name": config.display_name,
        "source_csv": str(source_path),
        "output_root": str(output_root),
        "folds": args.folds,
        "seed": args.seed,
        "splitter": splitter_name,
        "split_group_column": PATIENT_ID_COLUMN,
        "patient_id_source": patient_id_source_used,
        "patient_count": int(patient_groups.nunique()),
        "skip_fis": args.skip_fis,
        "skip_smote": args.skip_smote,
        "fold_records": fold_records,
    }
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("=" * 100)
    print(f"Prepared modality: {config.key}")
    print(f"Source: {source_path}")
    print(f"Splitter: {splitter_name} by {PATIENT_ID_COLUMN}; patients={patient_groups.nunique()}")
    print(f"Output: {output_root}")
    print(f"Manifest: {manifest_path}")
    print("=" * 100)
    return manifest


def main():
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    os.chdir(PROJECT_ROOT)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_root = resolve_project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "manifest_all.json"
    report_root = resolve_project_path(args.report_root)

    if args.report_only or args.only_fkg:
        if not summary_path.exists():
            raise FileNotFoundError(f"Existing manifest not found: {summary_path}")
        manifest_data = json.loads(summary_path.read_text(encoding="utf-8"))
        manifests = manifest_data["manifests"]
        if args.only_fkg:
            run_fkg_for_manifests(manifests, args.fkg_backend)
    else:
        manifests = []
        for modality in selected_modalities(args.modalities):
            manifests.append(prepare_modality(MODALITY_CONFIGS[modality], args))

    report_outputs = write_report_outputs(manifests, report_root)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "modalities": [manifest["modality"] for manifest in manifests],
                "report_outputs": report_outputs,
                "manifests": manifests,
            },
            file,
            indent=2,
        )
    print(f"[DONE] KFold Feature Selection manifest: {summary_path}")
    print(f"[DONE] KFold Feature Selection report: {report_outputs['summary_csv']}")


if __name__ == "__main__":
    main()
