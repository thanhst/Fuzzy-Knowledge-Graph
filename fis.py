"""Small NumPy implementation of the 1-D FCM/FIS rule generator."""

from __future__ import annotations

import numpy as np


class FIS:
    def __init__(self, clusters: int = 5, m: float = 2.0, eps: float = 1e-5, max_iter: int = 200):
        self.clusters = clusters
        self.m = m
        self.eps = eps
        self.max_iter = max_iter

    def _fcm(self, values: np.ndarray, count: int) -> np.ndarray:
        minimum, maximum = float(values.min()), float(values.max())
        centers = np.linspace(minimum, maximum, count, dtype=np.float64)
        previous = np.inf
        exponent = 2.0 / (self.m - 1.0)
        for _ in range(self.max_iter):
            distance = np.maximum(np.abs(values[:, None] - centers[None, :]), self.eps)
            ratios = (distance[:, :, None] / distance[:, None, :]) ** exponent
            membership = 1.0 / ratios.sum(axis=2)
            membership /= membership.sum(axis=1, keepdims=True)
            weights = membership**self.m
            updated = (weights * values[:, None]).sum(axis=0) / weights.sum(axis=0)
            objective = float((weights * distance**2).sum())
            centers = updated
            if abs(objective - previous) < self.eps:
                break
            previous = objective
        return centers

    def fit(self, train_features: np.ndarray) -> "FIS":
        values = np.asarray(train_features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("train_features must be a non-empty matrix")
        self.centers = [self._fcm(values[:, column], self.clusters) for column in range(values.shape[1])]
        self.sigmas = []
        for centers in self.centers:
            distance = float(np.max(centers) - np.min(centers))
            sigma = abs(distance) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            while sigma < 1.0:
                sigma *= 10.0
            self.sigmas.append(sigma)
        return self

    def fuzzify(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[1] != len(self.centers):
            raise ValueError("feature count does not match fitted FIS")
        rules = np.empty(values.shape, dtype=np.int64)
        for column, (centers, sigma) in enumerate(zip(self.centers, self.sigmas)):
            difference = values[:, column, None] - centers[None, :]
            membership = np.exp(-(difference * difference) / (2.0 * sigma * sigma))
            rules[:, column] = np.argmax(membership, axis=1) + 1
        return rules

    def make_rules(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        encoded_labels = np.asarray(labels, dtype=np.int64) + 1
        return np.column_stack([self.fuzzify(features), encoded_labels])
