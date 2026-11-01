"""Fold-local feature processing and BorderlineSMOTE with exact parent tracking.

Median imputation, standardization and ANOVA selection follow the source fusion
pipeline: 7 image and 9 clinical features, all fitted on inner training only.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from build import IMAGE_COLUMNS, TABLE_COLUMNS


class FusionPreprocessor:
    def __init__(self, k_img=7, k_tab=9):
        self.k_img, self.k_tab = k_img, k_tab

    def fit(self, frame, labels):
        self.branches = []
        self.feature_names = []
        for columns, k in ((IMAGE_COLUMNS, self.k_img), (TABLE_COLUMNS, self.k_tab)):
            values = frame[columns].replace([np.inf, -np.inf], np.nan)
            medians = values.median().fillna(0)
            scaler = StandardScaler().fit(values.fillna(medians))
            scaled = scaler.transform(values.fillna(medians))
            selector = SelectKBest(f_classif, k=min(k, len(columns))).fit(scaled, labels)
            self.branches.append((columns, medians, scaler, selector))
            self.feature_names.extend(np.array(columns)[selector.get_support()].tolist())
        selected = self._transform_all(frame)
        self.keep = np.ptp(selected, axis=0) > 0
        if self.keep.sum() < 2:
            raise ValueError("FKG-E needs at least two nonconstant selected features")
        self.feature_names = np.array(self.feature_names)[self.keep].tolist()
        return self

    def _transform_all(self, frame):
        results = []
        for columns, medians, scaler, selector in self.branches:
            values = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(medians)
            results.append(selector.transform(scaler.transform(values)))
        return np.column_stack(results)

    def transform(self, frame):
        return self._transform_all(frame)[:, self.keep]


class TrackedBorderlineSMOTE(BorderlineSMOTE):
    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps, y_type, y=None):
        if self.kind != "borderline-1":
            raise ValueError("Parent tracking supports borderline-1 only")
        minority = np.flatnonzero(self.original_y == y_type)
        left = self.in_danger_indices[y_type][rows]
        right = minority[nn_num[rows, cols]]
        self.parent_batches.append(np.column_stack([left, right]))
        return super()._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type, y)


def balance_training(X, y, patients, seed):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        raise ValueError("Inner train requires at least two images in each class")
    sampler = TrackedBorderlineSMOTE(random_state=seed, kind="borderline-1",
                                     k_neighbors=min(5, int(counts.min()) - 1),
                                     m_neighbors=min(10, len(y) - 1))
    sampler.original_y = y.copy()
    sampler.parent_batches = []
    resampled, target = sampler.fit_resample(X, y)
    pairs = np.vstack(sampler.parent_batches) if sampler.parent_batches else np.empty((0, 2), dtype=int)
    if len(resampled) != len(X) + len(pairs):
        raise AssertionError("SMOTE provenance does not cover every synthetic row")
    # Synthetic rows receive distinct placeholders, then are linked to BOTH
    # real parents by connected_units; they are never counted as real patients.
    groups = np.r_[np.asarray(patients, dtype=str), [f"synthetic:{i}" for i in range(len(pairs))]]
    return resampled, target, groups, pairs


def balance_teacher_parts(X, y, patients, seed):
    """Resample within pre-existing patient-disjoint teacher partitions only."""
    y, patients = np.asarray(y, int), np.asarray(patients, str)
    parts = np.zeros(len(y), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for part, (_, indices) in enumerate(splitter.split(X, y, groups=patients)):
        parts[indices] = part
    extra_x, extra_y, extra_parts, all_parents = [], [], [], []
    for part in range(5):
        indices = np.flatnonzero(parts == part)
        resampled, target, _, parents = balance_training(X[indices], y[indices], patients[indices], seed + part)
        extra_x.append(resampled[len(indices):])
        extra_y.append(target[len(indices):])
        extra_parts.extend([part] * len(parents))
        all_parents.append(indices[parents])
    Xnew = np.vstack([X, *extra_x])
    ynew = np.concatenate([y, *extra_y])
    parents = np.vstack(all_parents)
    groups = np.r_[patients, [f"synthetic:{i}" for i in range(len(parents))]]
    all_parts = np.r_[parts, extra_parts].astype(int)
    if not np.array_equal(all_parts[len(y):], parts[parents[:, 0]]) or not np.array_equal(
        all_parts[len(y):], parts[parents[:, 1]]
    ):
        raise AssertionError("A synthetic row crosses its parents' teacher partitions")
    return Xnew, ynew, groups, parents, all_parts
