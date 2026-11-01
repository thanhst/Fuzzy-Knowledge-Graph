"""Patient-level split helpers requiring only NumPy, pandas and scikit-learn."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def clean_records(frame, label_column):
    required = ["image_id", "patient_id", label_column]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame[required].copy()
    audit = {"input_rows": len(data)}
    for column in ("image_id", "patient_id"):
        data[column] = data[column].astype("string").str.strip()
    invalid = data[["image_id", "patient_id"]].isna().any(axis=1)
    invalid |= data[["image_id", "patient_id"]].eq("").any(axis=1)
    audit["removed_missing_ids"] = int(invalid.sum())
    data = data.loc[~invalid].copy()
    duplicate = data.image_id.duplicated(keep=False)
    audit["removed_duplicate_image_ids"] = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    labels = pd.to_numeric(data[label_column], errors="coerce")
    valid = labels.isin([0, 1])
    audit["removed_invalid_binary_labels"] = int((~valid).sum())
    data = data.loc[valid].copy()
    data[label_column] = labels.loc[valid].astype(int)
    data = data.reset_index(drop=True)
    audit["retained_images"] = len(data)
    audit["retained_patients"] = int(data.patient_id.nunique())
    if data.empty:
        raise ValueError("No valid image records remain after filtering")
    return data, audit


def create_assignments(data, label_column, method="stratified-group", folds=5, seed=42):
    if method != "stratified-group":
        raise ValueError("Only StratifiedGroupKFold is supported")
    if folds < 2 or data.patient_id.nunique() < folds:
        raise ValueError("Need at least two folds and at least one patient per fold")
    assigned = data.reset_index(drop=True).copy()
    assigned["fold"] = 0
    appearances = np.zeros(len(assigned), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(
        np.zeros((len(assigned), 1)), assigned[label_column], groups=assigned.patient_id
    ), 1):
        assigned.loc[test, "fold"] = fold
        appearances[test] += 1
    if not (appearances == 1).all():
        raise AssertionError("Each image must appear in held-out exactly once")
    return assigned, "StratifiedGroupKFold"


def verify_assignments(assigned, label_column, folds):
    clean, _ = clean_records(assigned, label_column)
    if len(clean) != len(assigned):
        raise AssertionError("Invalid or duplicate IDs/labels in assignments")
    if not assigned["fold"].isin(range(1, folds + 1)).all():
        raise AssertionError("Unassigned image or invalid fold number")
    if (assigned.groupby("patient_id")["fold"].nunique() != 1).any():
        raise AssertionError("A patient's images cross held-out folds")
    details = []
    for fold in range(1, folds + 1):
        test = assigned.loc[assigned["fold"] == fold]
        train = assigned.loc[assigned["fold"] != fold]
        overlap = set(train.patient_id) & set(test.patient_id)
        if test.empty or train.empty or overlap:
            raise AssertionError(f"Invalid or leaking fold {fold}")
        details.append({"fold": fold, "train_images": len(train), "test_images": len(test),
                        "train_patients": int(train.patient_id.nunique()),
                        "test_patients": int(test.patient_id.nunique()),
                        "test_positives": int(test[label_column].sum()),
                        "patient_overlap_count": len(overlap)})
    return {"each_image_held_out_once": True, "each_patient_in_one_held_out_fold": True,
            "patient_overlap_count": 0, "folds": details}
