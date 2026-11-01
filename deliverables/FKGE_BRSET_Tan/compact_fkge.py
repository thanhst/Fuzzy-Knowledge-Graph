"""Memory-bounded FISA-GM/FKG-E, preserving duplicate-rule multiplicities.

All numerical operations use float64. Duplicate crisp rules have identical
embeddings and attributions; aggregating their probabilities with multiplicity
preserves the original log-sum-exp, KL losses and full-batch gradient.
"""
from __future__ import annotations

import math
import time
from itertools import combinations

import numpy as np
from scipy import sparse


class FCMFuzzifier:
    def __init__(self, n_clusters=5, m=2.0, iters=150, tol=1e-6):
        self.c, self.m, self.iters, self.tol = n_clusters, m, iters, tol

    def fit(self, X):
        self.centers = []
        for x in np.asarray(X).T:
            c = min(self.c, len(np.unique(x)))
            v = np.quantile(x, (np.arange(c) + 0.5) / c)
            for _ in range(self.iters):
                d = np.abs(x[:, None] - v[None, :]) + 1e-12
                U = 1 / ((d[:, :, None] / d[:, None, :]) ** (2 / (self.m - 1))).sum(2)
                Um = U ** self.m
                new = (Um * x[:, None]).sum(0) / Um.sum(0)
                done = np.max(np.abs(new - v)) < self.tol
                v = new
                if done:
                    break
            self.centers.append(np.sort(v))
        return self

    def transform(self, X):
        return np.column_stack([np.argmin(np.abs(X[:, j, None] - v[None, :]), axis=1)
                                for j, v in enumerate(self.centers)])


def connected_units(patients, signatures, parent_pairs=(), merge_signatures=True):
    parent = list(range(len(patients)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # With known real patient IDs, equal fuzzy patterns in different patients
    # do not make those patients the same subject. Legacy comparison may merge.
    for keys in ((patients, signatures) if merge_signatures else (patients,)):
        first = {}
        for i, key in enumerate(keys):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    # A synthetic row and both of its real SMOTE parents stay in one teacher unit.
    original_n = len(patients) - len(parent_pairs)
    for offset, (left, right) in enumerate(parent_pairs):
        child = original_n + offset
        for source in (int(left), int(right)):
            parent[find(child)] = find(source)
    mapping = {}
    return np.array([mapping.setdefault(find(i), len(mapping)) for i in range(len(parent))])


def unit_partition(units, n_parts, seed, y):
    unique, inverse = np.unique(units, return_inverse=True)
    classes = np.unique(y)
    counts = np.zeros((len(unique), len(classes)))
    np.add.at(counts, (inverse, np.searchsorted(classes, y)), 1)
    total = counts.sum(0)
    order = np.random.RandomState(seed).permutation(len(unique))
    order = order[np.argsort(-counts[order].sum(1), kind="stable")]
    load = np.zeros((n_parts, len(classes)))
    assignment = np.zeros(len(unique), dtype=int)
    for k in order:
        fill = (load / total) @ (counts[k] / counts[k].sum())
        p = int(np.lexsort((load.sum(1), fill))[0])
        assignment[k] = p
        load[p] += counts[k]
    return assignment[inverse]


def compress_rules(R, y, patients, parent_pairs=(), merge_signatures=True):
    full = np.column_stack([R, y])
    _, sig = np.unique(full, axis=0, return_inverse=True)
    units = connected_units(patients, sig, parent_pairs, merge_signatures)
    values, index, inverse, counts = np.unique(np.column_stack([full, units]), axis=0,
                                              return_index=True, return_inverse=True, return_counts=True)
    return {"R": values[:, :-2].astype(int), "y": values[:, -2].astype(int),
            "units": values[:, -1].astype(int), "weights": counts.astype(float),
            "index": index, "inverse": inverse, "original_units": units}


def threshold_on_validation(y, scores):
    candidates = np.unique(np.r_[0.0, np.quantile(scores, np.linspace(0.01, 0.99, 197))])
    best, threshold = -1.0, 0.0
    for t in candidates:
        value = 0.5 * (np.mean(scores[y == 1] > t) + np.mean(scores[y == 0] <= t))
        if value > best + 1e-12:
            best, threshold = value, float(t)
    return threshold


class CompactFISA:
    def __init__(self, R, y, weights, levels=5):
        started = time.perf_counter()
        self.R, self.y, self.weights = np.asarray(R, int), np.asarray(y, int), np.asarray(weights, float)
        self.levels = levels
        n, m = self.R.shape
        total = self.weights.sum()
        self.nodes = np.array(list(combinations(range(m), min(3, m))))
        self.n_keys = levels ** self.nodes.shape[1]
        sum_A = np.zeros(n)
        for columns in combinations(range(m), min(4, m)):
            key = np.zeros(n, dtype=int)
            for j in columns:
                key = key * levels + self.R[:, j]
            count = np.bincount(key, weights=self.weights, minlength=levels ** len(columns))
            sum_A += count[key] / total
        M = np.empty((n, m))
        for j in range(m):
            key = self.R[:, j] * 2 + self.y
            count = np.bincount(key, weights=self.weights, minlength=levels * 2)
            M[:, j] = count[key] / total
        self.B = np.empty((n, len(self.nodes)))
        self.keys = self._keys(self.R)
        self.W = np.zeros((len(self.nodes), self.n_keys, 2))
        for p, columns in enumerate(self.nodes):
            self.B[:, p] = sum_A * M[:, columns].min(1)
            np.add.at(self.W[p], (self.keys[:, p], self.y), self.weights * self.B[:, p])
        self.fallback = int(np.argmax(np.bincount(self.y, weights=self.weights, minlength=2)))
        self.fit_seconds = time.perf_counter() - started
        self._attribution_matrices = {}

    def _keys(self, R):
        keys = np.zeros((len(R), len(self.nodes)), dtype=np.int32)
        for j in range(self.nodes.shape[1]):
            keys = keys * self.levels + R[:, self.nodes[:, j]]
        return keys

    def log_values(self, R, batch=256):
        out, matched = [], []
        p = np.arange(len(self.nodes))[None, :]
        for start in range(0, len(R), batch):
            keys = self._keys(R[start:start + batch])
            V = self.W[p, keys]
            log_D = np.log(V + 1e-9).mean(1)
            out.append(np.logaddexp(log_D, np.log(1e-12)))
            matched.append((V.sum(2) > 0).any(1))
        return np.concatenate(out), np.concatenate(matched)

    def attribution(self, R, chosen_class, batch=64):
        n, n_rules = len(R), len(self.R)
        result = np.zeros((n, n_rules))
        node_ids = np.arange(len(self.nodes))[None, :]
        for label in (0, 1):
            ridx = np.flatnonzero(self.y == label)
            qidx = np.flatnonzero(chosen_class == label)
            if not len(ridx) or not len(qidx):
                continue
            if label not in self._attribution_matrices:
                columns = (self.keys[ridx] + node_ids * self.n_keys).astype(np.int32)
                values = self.B[ridx] * self.weights[ridx, None]
                offsets = np.arange(0, values.size + 1, len(self.nodes), dtype=np.int32)
                self._attribution_matrices[label] = sparse.csr_matrix(
                    (values.ravel(), columns.ravel(), offsets),
                    shape=(len(ridx), len(self.nodes) * self.n_keys))
            rule_matrix = self._attribution_matrices[label]
            for start in range(0, len(qidx), batch):
                indices = qidx[start:start + batch]
                keys = self._keys(R[indices])
                V = self.W[node_ids, keys, label]
                active = V > 0
                count = active.sum(1)
                values = np.divide(1.0, V * np.maximum(count, 1)[:, None],
                                   out=np.zeros_like(V), where=active)
                columns = (keys + node_ids * self.n_keys).astype(np.int32)
                offsets = np.arange(0, values.size + 1, len(self.nodes), dtype=np.int32)
                query_matrix = sparse.csr_matrix((values.ravel(), columns.ravel(), offsets),
                                                 shape=(len(indices), rule_matrix.shape[1]))
                a = (query_matrix @ rule_matrix.T).toarray()
                totals = a.sum(1)
                fallback = totals <= 0
                a[~fallback] /= totals[~fallback, None]
                a[fallback] = self.weights[ridx] / self.weights[ridx].sum()
                result[np.ix_(indices, ridx)] = a
        return result
