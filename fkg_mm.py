"""Small, platform-independent reference implementation of native FKG-MM.

The input is an FIS fuzzy-rule table: integer-valued feature rules followed by
the 1-based class label.  The implementation preserves the algorithm used by
the native C++ runner, including its four-feature A relation, three-feature
B/C relations, column-wise min-max normalization, and legacy omission of the
last training row during inference lookup.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


class FKGMM:
    """Train and evaluate FKG-MM from already generated multimodal FRB rules."""

    def fit(self, rules: np.ndarray) -> "FKGMM":
        base = np.asarray(rules, dtype=np.int64)
        if base.ndim != 2 or base.shape[0] < 2 or base.shape[1] < 5:
            raise ValueError("rules must contain at least two rows and four features plus label")

        features, labels = base[:, :-1], base[:, -1]
        if labels.min() < 1:
            raise ValueError("class labels must be 1-based positive integers")

        self.feature_count = features.shape[1]
        self.class_count = max(int(labels.max()), len(np.unique(labels)))
        self.offset = int(features.min())
        self.radix = int(features.max() - self.offset + 1)
        self.comb3 = np.asarray(list(combinations(range(self.feature_count), 3)))
        comb4 = combinations(range(self.feature_count), 4)
        row_count = features.shape[0]

        # A[r, q] is the frequency of row r's four-rule pattern.  Only sum(A[r])
        # is required by B, so the large A matrix need not be retained.
        sum_a = np.zeros(row_count, dtype=np.float64)
        for columns in comb4:
            codes = self._encode(features, columns)
            _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
            sum_a += counts[inverse] / row_count

        # M[r, f] is the frequency of the (feature rule, class) pair.
        m = np.empty(features.shape, dtype=np.float64)
        label_stride = self.class_count + 1
        for feature_index in range(self.feature_count):
            pair_codes = (
                (features[:, feature_index] - self.offset) * label_stride + labels
            )
            _, inverse, counts = np.unique(
                pair_codes, return_inverse=True, return_counts=True
            )
            m[:, feature_index] = counts[inverse] / row_count

        code_count = self.radix**3
        tables = np.zeros(
            (len(self.comb3), self.class_count, code_count), dtype=np.float64
        )

        for combination_index, columns in enumerate(self.comb3):
            triple_codes = self._encode(features, columns)
            b = sum_a * np.min(m[:, columns], axis=1)

            # C aggregates B for every (three-rule pattern, class) pair.
            group_codes = triple_codes * label_stride + labels
            totals = np.bincount(
                group_codes,
                weights=b,
                minlength=code_count * label_stride,
            )

            for label in range(1, self.class_count + 1):
                raw_by_code = totals[label::label_stride][:code_count]
                raw_rows = raw_by_code[triple_codes]
                minimum, maximum = float(raw_rows.min()), float(raw_rows.max())
                normalized = np.zeros(code_count, dtype=np.float64)
                if maximum > minimum:
                    normalized = (raw_by_code - minimum) / (maximum - minimum)

                # The published native implementation scans base[0:-1] during
                # FISA.  Keep that behavior so reviewer predictions are exact.
                present = np.unique(triple_codes[:-1][labels[:-1] == label])
                tables[combination_index, label - 1, present] = normalized[present]

        self._tables = tables
        return self

    def predict_batch_with_confidence(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "_tables"):
            raise RuntimeError("fit must be called before predict")

        samples = np.asarray(features, dtype=np.int64)
        if samples.ndim != 2 or samples.shape[1] != self.feature_count:
            raise ValueError(f"expected a matrix with {self.feature_count} features")
        if samples.min() < self.offset or samples.max() >= self.offset + self.radix:
            raise ValueError("test rules contain a value outside the training rule domain")

        predictions = np.empty(samples.shape[0], dtype=np.int64)
        confidences = np.empty(samples.shape[0], dtype=np.float64)
        combination_indices = np.arange(len(self.comb3))

        for row_index, sample in enumerate(samples):
            selected = sample[self.comb3] - self.offset
            codes = (selected[:, 0] * self.radix + selected[:, 1]) * self.radix + selected[:, 2]
            values = self._tables[combination_indices, :, codes]
            d_values = values.max(axis=0) + values.min(axis=0)
            winner = int(np.argmax(d_values))
            predictions[row_index] = winner + 1
            total = float(d_values.sum())
            confidences[row_index] = float(d_values[winner] / total) if total > 0 else 0.0

        return predictions, confidences

    def _encode(self, features: np.ndarray, columns) -> np.ndarray:
        selected = features[:, columns] - self.offset
        codes = np.zeros(features.shape[0], dtype=np.int64)
        for column in range(selected.shape[1]):
            codes = codes * self.radix + selected[:, column]
        return codes
