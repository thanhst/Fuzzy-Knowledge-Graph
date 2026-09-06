import re

import numpy as np
import pandas as pd


def _group_size_summary(groups):
    sizes = pd.Series(groups).value_counts(sort=False)
    return {
        "group_count": int(sizes.shape[0]),
        "group_size_min": int(sizes.min()) if not sizes.empty else 0,
        "group_size_max": int(sizes.max()) if not sizes.empty else 0,
    }


def _row_pair_groups(row_count, images_per_patient):
    if images_per_patient < 1:
        raise ValueError("images_per_patient must be at least 1.")
    return pd.Series(
        [f"row_pair_{idx // images_per_patient}" for idx in range(row_count)],
        name="patient_group",
    )


def _numeric_image_pair_groups(values, images_per_patient, source_column):
    as_text = pd.Series(values).astype(str)
    numbers = as_text.str.extract(r"(\d+)$", expand=False)
    if numbers.isna().any():
        return None

    numeric_values = numbers.astype(int)
    groups = ((numeric_values - 1) // images_per_patient).astype(str)
    return pd.Series(
        source_column + "_pair_" + groups,
        name="patient_group",
    )


def build_patient_groups(df, images_per_patient=2):
    if images_per_patient < 1:
        raise ValueError("images_per_patient must be at least 1.")

    for column in ("patient_id", "patient", "subject_id"):
        if column in df.columns:
            groups = pd.Series(column + "_" + df[column].astype(str), name="patient_group")
            metadata = {
                "patient_group_source": column,
                "patient_images_per_group": int(images_per_patient),
            }
            metadata.update(_group_size_summary(groups))
            return groups.reset_index(drop=True), metadata

    for column in ("image_id", "img_id", "id"):
        if column not in df.columns:
            continue

        groups = _numeric_image_pair_groups(
            df[column],
            images_per_patient=images_per_patient,
            source_column=column,
        )
        if groups is not None:
            metadata = {
                "patient_group_source": f"{column}_numeric_pair",
                "patient_images_per_group": int(images_per_patient),
            }
            metadata.update(_group_size_summary(groups))
            return groups.reset_index(drop=True), metadata

        normalized = (
            df[column]
            .astype(str)
            .str.replace(re.compile(r"([_-]?(left|right|os|od|le|re))$"), "", regex=True)
        )
        if normalized.duplicated(keep=False).any():
            groups = pd.Series(column + "_" + normalized, name="patient_group")
            metadata = {
                "patient_group_source": f"{column}_normalized_eye_suffix",
                "patient_images_per_group": int(images_per_patient),
            }
            metadata.update(_group_size_summary(groups))
            return groups.reset_index(drop=True), metadata

    groups = _row_pair_groups(len(df), images_per_patient)
    metadata = {
        "patient_group_source": "row_order_pair",
        "patient_images_per_group": int(images_per_patient),
    }
    metadata.update(_group_size_summary(groups))
    return groups.reset_index(drop=True), metadata


def group_kfold_split(groups, n_splits, seed):
    groups = pd.Series(groups).reset_index(drop=True)
    unique_groups = groups.drop_duplicates().to_numpy()
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if unique_groups.shape[0] < n_splits:
        raise ValueError(
            f"Need at least {n_splits} patient groups for {n_splits} folds, "
            f"got {unique_groups.shape[0]}."
        )

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    fold_groups = np.array_split(shuffled_groups, n_splits)
    positions = np.arange(groups.shape[0])

    for test_groups in fold_groups:
        test_mask = groups.isin(test_groups).to_numpy()
        yield positions[~test_mask], positions[test_mask]


def group_train_test_split(df, test_size=0.3, seed=None, images_per_patient=2):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    groups, group_metadata = build_patient_groups(
        df,
        images_per_patient=images_per_patient,
    )
    unique_groups = groups.drop_duplicates().to_numpy()
    if unique_groups.shape[0] < 2:
        raise ValueError("Need at least 2 patient groups for train/test split.")

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    test_group_count = int(round(unique_groups.shape[0] * test_size))
    test_group_count = min(max(test_group_count, 1), unique_groups.shape[0] - 1)

    test_groups = set(shuffled_groups[:test_group_count])
    test_mask = groups.isin(test_groups).to_numpy()
    train_df = df.loc[~test_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)

    split_summary = split_group_summary(
        groups.loc[~test_mask],
        groups.loc[test_mask],
    )
    group_metadata.update(split_summary)
    return train_df, test_df, group_metadata


def split_group_summary(train_groups, test_groups):
    train_set = set(pd.Series(train_groups).astype(str))
    test_set = set(pd.Series(test_groups).astype(str))
    overlap = sorted(train_set & test_set)
    return {
        "train_group_count": int(len(train_set)),
        "test_group_count": int(len(test_set)),
        "patient_group_overlap_count": int(len(overlap)),
        "patient_group_overlap_sample": overlap[:10],
    }
