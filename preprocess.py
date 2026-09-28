"""Leakage-safe fold preparation used by the FKG-MM experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler


LABEL = "diabetic_retinopathy"
IMAGE_FEATURES = [
    "Contrast Feature", "Dissimilarity Feature", "Homogeneity Feature",
    "Energy Feature", "Correlation Feature", "ASM Feature", "Mean Feature",
    "Variance Feature", "Standard Deviation Feature", "RMS Feature",
]
TABLE_FEATURES = [
    "patient_age", "patient_sex", "diabetes_time_y", "insuline", "diabetes",
    "exam_eye", "optic_disc", "vessels", "macula", "focus", "Illuminaton",
    "image_field", "quality",
]


def _read_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"image_id": str})
    values = frame["image_id"].astype(str).str.strip()
    if values.duplicated().any():
        raise ValueError(f"duplicate image_id in {path}")
    return sorted(values.tolist())


def load_source(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    source = pd.read_csv(data_dir / "fusion_features.csv")
    sidecar = pd.read_csv(data_dir / "row_ids.csv", dtype={"image_id": str})
    labels = pd.read_csv(
        data_dir / "labels_brset.csv", dtype={"image_id": str, "patient_id": str}
    )[["image_id", "patient_id"]]
    if len(source) != len(sidecar):
        raise ValueError("fusion_features.csv and row_ids.csv are not row-aligned")

    ids = sidecar[["image_id"]].merge(labels, on="image_id", how="left", validate="one_to_one")
    if ids["patient_id"].isna().any():
        raise ValueError("some feature rows have no patient_id")
    raw_labels = source.pop(LABEL).astype(str)
    encoded = pd.Series(LabelEncoder().fit_transform(raw_labels), name=LABEL)
    features = source.apply(pd.to_numeric, errors="coerce")
    return features, encoded, ids


def _scale(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = train.median(numeric_only=True).fillna(0.0)
    train = train.fillna(medians)
    test = test.fillna(medians)
    scaler = StandardScaler()
    return (
        pd.DataFrame(scaler.fit_transform(train), columns=train.columns),
        pd.DataFrame(scaler.transform(test), columns=test.columns),
    )


def _select(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: pd.Series,
    k: int,
    branch: str,
    output_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    selector = SelectKBest(f_classif, k=min(k, train.shape[1]))
    train_selected = selector.fit_transform(train, labels)
    test_selected = selector.transform(test)
    chosen = list(train.columns[selector.get_support()])
    output_positions = {name: output_start + pos for pos, name in enumerate(chosen)}
    rows = []
    for index, name in enumerate(train.columns):
        rows.append({
            "branch": branch,
            "selected": name in output_positions,
            "source_column": name,
            "selected_output_column": output_positions.get(name, ""),
            "score": float(selector.scores_[index]),
            "p_value": float(selector.pvalues_[index]),
        })
    return pd.DataFrame(train_selected), pd.DataFrame(test_selected), rows


def prepare_fold(
    features: pd.DataFrame,
    labels: pd.Series,
    ids: pd.DataFrame,
    split_dir: Path,
    fold_number: int,
    seed: int = 42,
) -> dict:
    fold_dir = split_dir / "train_kfold" / f"fold_{fold_number}"
    train_ids = _read_ids(fold_dir / "train.csv")
    test_ids = _read_ids(fold_dir / "val.csv")
    if set(train_ids) & set(test_ids):
        raise RuntimeError(f"image leakage in fold {fold_number}")

    index_by_id = {image_id: index for index, image_id in enumerate(ids["image_id"])}
    missing = (set(train_ids) | set(test_ids)) - set(index_by_id)
    if missing:
        raise ValueError(f"fold {fold_number} contains unknown image ids: {sorted(missing)[:5]}")
    train_index = np.asarray([index_by_id[value] for value in train_ids])
    test_index = np.asarray([index_by_id[value] for value in test_ids])

    train_patients = set(ids.iloc[train_index]["patient_id"])
    test_patients = set(ids.iloc[test_index]["patient_id"])
    overlap = train_patients & test_patients
    if overlap:
        raise RuntimeError(f"patient leakage in fold {fold_number}: {sorted(overlap)[:5]}")

    train_x, test_x = features.iloc[train_index], features.iloc[test_index]
    train_y = labels.iloc[train_index].reset_index(drop=True)
    test_y = labels.iloc[test_index].reset_index(drop=True)
    train_img, test_img = _scale(train_x[IMAGE_FEATURES], test_x[IMAGE_FEATURES])
    train_tab, test_tab = _scale(train_x[TABLE_FEATURES], test_x[TABLE_FEATURES])
    train_img, test_img, image_rows = _select(train_img, test_img, train_y, 7, "image", 0)
    train_tab, test_tab, table_rows = _select(train_tab, test_tab, train_y, 9, "table", 7)

    train_selected = pd.concat([train_img, train_tab], axis=1, ignore_index=True)
    test_selected = pd.concat([test_img, test_tab], axis=1, ignore_index=True)
    keep = [i for i in range(train_selected.shape[1]) if train_selected.iloc[:, i].nunique() > 1]
    if len(keep) != train_selected.shape[1]:
        train_selected, test_selected = train_selected.iloc[:, keep], test_selected.iloc[:, keep]

    train_frame = train_selected.reset_index(drop=True)
    train_frame[LABEL] = train_y
    test_frame = test_selected.reset_index(drop=True)
    test_frame[LABEL] = test_y
    sampler = BorderlineSMOTE(random_state=seed + fold_number, k_neighbors=5)
    x_resampled, y_resampled = sampler.fit_resample(
        train_frame.drop(columns=LABEL), train_frame[LABEL]
    )
    train_frame = pd.DataFrame(x_resampled)
    train_frame.columns = [str(i) for i in range(train_frame.shape[1])]
    train_frame[LABEL] = pd.Series(y_resampled).reset_index(drop=True)
    test_frame.columns = [str(i) for i in range(test_frame.shape[1] - 1)] + [LABEL]

    return {
        "train": train_frame,
        "test": test_frame,
        "train_ids": ids.iloc[train_index].reset_index(drop=True),
        "test_ids": ids.iloc[test_index].reset_index(drop=True),
        "selected_features": pd.DataFrame(image_rows + table_rows),
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
    }
