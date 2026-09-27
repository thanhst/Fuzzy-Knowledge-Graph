"""
models/kge_baselines.py — Ba baseline đối chứng cho FKG-E (Mục 3.5.4):
  1. Node2Vec-lite  : random walk trên đồ thị đồng xuất hiện + skip-gram
  2. TransE-lite    : h + r ~ t, margin ranking loss
  3. DistMult-lite  : h^T diag(r) t, margin ranking loss

Cả ba đều KHÔNG dùng cấu trúc luật IF-THEN (khác FKG-E) — chỉ học từ đồ thị
đồng xuất hiện (node2vec) hoặc từ các "triple" suy ra từ luật (TransE/
DistMult). Downstream classifier dùng chung: k-Nearest-Neighbor trên không
gian nhúng (đúng thiết kế "Node2Vec + kNN", "TransE + kNN" trong tài liệu
thiết kế thực nghiệm).
"""
import numpy as np
import random
import time
from collections import defaultdict, Counter


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ============================================================
# 1) NODE2VEC-LITE
# ============================================================
class Node2VecLite:
    def __init__(self, fkg, d=32, walk_len=20, n_walks=10, window=2,
                 K_neg=5, lr=0.05, epochs=20, seed=42):
        self.fkg = fkg
        self.vocab = fkg.vocab
        self.token2idx = fkg.token2idx
        self.V = len(self.vocab)
        self.d = d
        self.walk_len, self.n_walks, self.window = walk_len, n_walks, window
        self.K_neg, self.lr, self.epochs = K_neg, lr, epochs
        rng = np.random.RandomState(seed)
        self.E = rng.normal(0, 0.1, size=(self.V, d))
        self.rng_py = random.Random(seed)
        self._build_graph()

    def _build_graph(self):
        adj = defaultdict(list)
        for e in self.fkg.edges:
            if e["u"] in self.token2idx and e["v"] in self.token2idx:
                u, v = self.token2idx[e["u"]], self.token2idx[e["v"]]
                adj[u].append(v)
                adj[v].append(u)
        self.adj = adj
        nodes_with_edges = list(adj.keys())
        self.nodes_with_edges = nodes_with_edges if nodes_with_edges else list(range(self.V))

    def _random_walks(self):
        walks = []
        for _ in range(self.n_walks):
            for start in self.nodes_with_edges:
                walk = [start]
                cur = start
                for _ in range(self.walk_len - 1):
                    nbrs = self.adj.get(cur, [])
                    if not nbrs:
                        break
                    cur = self.rng_py.choice(nbrs)
                    walk.append(cur)
                walks.append(walk)
        return walks

    def fit(self):
        t0 = time.time()
        walks = self._random_walks()
        pairs = []
        for walk in walks:
            n = len(walk)
            for t in range(n):
                lo, hi = max(0, t - self.window), min(n, t + self.window + 1)
                for c in range(lo, hi):
                    if c != t:
                        pairs.append((walk[t], walk[c]))
        freq = Counter([i for w in walks for i in w])
        freq_arr = np.ones(self.V) * 1e-6
        for i, c in freq.items():
            freq_arr[i] = c
        neg_dist = freq_arr ** 0.75
        neg_dist /= neg_dist.sum()

        for epoch in range(self.epochs):
            random.shuffle(pairs)
            for (i, j) in pairs[:2000]:  # giới hạn/epoch để tốc độ chấp nhận được
                zi, zj = self.E[i], self.E[j]
                sig = sigmoid(zi @ zj)
                g = (sig - 1.0) * self.lr
                self.E[i] -= g * zj
                self.E[j] -= g * zi
                negs = np.random.choice(self.V, size=self.K_neg, p=neg_dist)
                for n in negs:
                    zn = self.E[n]
                    sign = sigmoid(zi @ zn)
                    gn = sign * self.lr
                    self.E[i] -= gn * zn
                    self.E[n] -= gn * zi
        self.train_time_s = time.time() - t0
        return self


# ============================================================
# 2) TransE-lite / 3) DistMult-lite (dùng chung khung triple)
# ============================================================
def _build_triples(fkg, token2idx):
    """Suy triple (head, rel, tail) từ luật: mỗi cặp tiền đề đồng xuất hiện
    trong cùng luật -> quan hệ 'co_occurs'; mỗi tiền đề -> hệ quả -> quan hệ
    'implies'. Hai loại quan hệ là đủ để TransE/DistMult có cấu trúc học."""
    triples = []
    for r in fkg.rules:
        ante = [token2idx[t] for t in r["antecedent_tokens"] if t in token2idx]
        cons = token2idx.get(r["consequent_token"])
        for i in range(len(ante)):
            for j in range(len(ante)):
                if i != j:
                    triples.append((ante[i], 0, ante[j]))  # rel 0 = co_occurs
            if cons is not None:
                triples.append((ante[i], 1, cons))          # rel 1 = implies
    return triples


class _KGEBase:
    REL_NAMES = ["co_occurs", "implies"]

    def __init__(self, fkg, d=32, lr=0.02, epochs=30, margin=1.0, seed=42):
        self.fkg = fkg
        self.vocab = fkg.vocab
        self.token2idx = fkg.token2idx
        self.V = len(self.vocab)
        self.d = d
        self.lr, self.epochs, self.margin = lr, epochs, margin
        rng = np.random.RandomState(seed)
        self.E = rng.normal(0, 0.1, size=(self.V, d))
        self.R = rng.normal(0, 0.1, size=(len(self.REL_NAMES), d))
        self.triples = _build_triples(fkg, self.token2idx)

    def _score(self, h, r, t):
        raise NotImplementedError

    def _grad_step(self, h, r, t, h2, t2):
        raise NotImplementedError

    def fit(self):
        t0 = time.time()
        for epoch in range(self.epochs):
            random.shuffle(self.triples)
            for (h, r, t) in self.triples[:2000]:
                h2 = random.randrange(self.V)
                t2 = random.randrange(self.V)
                self._grad_step(h, r, t, h2, t2)
        self.train_time_s = time.time() - t0
        return self


class TransELite(_KGEBase):
    def fit(self):
        t0 = time.time()
        for epoch in range(self.epochs):
            random.shuffle(self.triples)
            for (h, r, t) in self.triples[:2000]:
                h2 = random.randrange(self.V)
                pos = self.E[h] + self.R[r] - self.E[t]
                neg = self.E[h2] + self.R[r] - self.E[t]
                pos_d = np.linalg.norm(pos) + 1e-9
                neg_d = np.linalg.norm(neg) + 1e-9
                loss = self.margin + pos_d - neg_d
                if loss > 0:
                    gpos = pos / pos_d
                    gneg = neg / neg_d
                    self.E[h] -= self.lr * gpos
                    self.E[t] += self.lr * gpos
                    self.R[r] -= self.lr * (gpos - gneg)
                    self.E[h2] -= self.lr * (-gneg)
        self.train_time_s = time.time() - t0
        return self

    def _score(self, h, r, t):
        return -np.linalg.norm(self.E[h] + self.R[r] - self.E[t])


class DistMultLite(_KGEBase):
    def fit(self):
        t0 = time.time()
        for epoch in range(self.epochs):
            random.shuffle(self.triples)
            for (h, r, t) in self.triples[:2000]:
                t2 = random.randrange(self.V)
                pos = np.sum(self.E[h] * self.R[r] * self.E[t])
                neg = np.sum(self.E[h] * self.R[r] * self.E[t2])
                sig_pos, sig_neg = sigmoid(pos), sigmoid(neg)
                # log-sigmoid ranking: muốn sig_pos lớn, sig_neg nhỏ
                g_pos = (sig_pos - 1.0) * self.lr
                g_neg = sig_neg * self.lr
                self.E[h] -= g_pos * (self.R[r] * self.E[t])
                self.E[t] -= g_pos * (self.R[r] * self.E[h])
                self.R[r] -= g_pos * (self.E[h] * self.E[t])
                self.E[h] -= g_neg * (self.R[r] * self.E[t2])
                self.E[t2] -= g_neg * (self.R[r] * self.E[h])
                self.R[r] -= g_neg * (self.E[h] * self.E[t2])
        self.train_time_s = time.time() - t0
        return self

    def _score(self, h, r, t):
        return float(np.sum(self.E[h] * self.R[r] * self.E[t]))


# ============================================================
# Downstream: kNN classifier trên không gian nhúng chung
# ============================================================
class KNNOnEmbedding:
    """Bọc ngoài (Node2Vec|TransE|DistMult) bằng một bộ phân loại kNN đơn
    giản: nhúng truy vấn = trung bình có trọng số các token active; nhãn dự
    đoán = nhãn phổ biến nhất trong k luật có nhúng gần nhất (đo bằng cosine)."""

    def __init__(self, fkg, embedding_table, token2idx, k=5):
        self.fkg = fkg
        self.E = embedding_table
        self.token2idx = token2idx
        self.k = k
        self.class_tokens = fkg.class_tokens
        self._prepare_rule_vectors()

    def _prepare_rule_vectors(self):
        vecs, labels = [], []
        for r in self.fkg.rules:
            ids = [self.token2idx[t] for t in r["antecedent_tokens"] if t in self.token2idx]
            if not ids:
                continue
            vecs.append(self.E[ids].mean(axis=0))
            labels.append(r["consequent_token"])
        self.rule_vecs = np.array(vecs)
        self.rule_labels = labels
        norms = np.linalg.norm(self.rule_vecs, axis=1, keepdims=True) + 1e-9
        self.rule_vecs_norm = self.rule_vecs / norms

    def _query_vec(self, membership):
        items = [(self.token2idx[t], deg) for t, deg in membership.items()
                  if t in self.token2idx and deg > 1e-8]
        if not items:
            return np.zeros(self.E.shape[1])
        ids = np.array([i for i, _ in items])
        degs = np.array([g for _, g in items])
        return (self.E[ids] * degs[:, None]).sum(axis=0) / (degs.sum() + 1e-9)

    def predict_one(self, membership):
        t0 = time.time()
        q = self._query_vec(membership)
        qn = q / (np.linalg.norm(q) + 1e-9)
        sims = self.rule_vecs_norm @ qn
        topk_idx = np.argsort(-sims)[:self.k]
        votes = Counter([self.rule_labels[i] for i in topk_idx])
        y_hat = votes.most_common(1)[0][0] if votes else self.class_tokens[0]
        elapsed = time.time() - t0
        # phân phối xác suất thô từ tỉ lệ phiếu bầu (dùng cho AUC nếu cần)
        proba = {c: votes.get(c, 0) / max(1, sum(votes.values())) for c in self.class_tokens}
        return proba, y_hat, elapsed

    def evaluate(self, samples):
        from models.metrics import classification_metrics
        y_true, y_pred, probabilities = [], [], []
        total_time = 0.0
        for s in samples:
            proba, y_hat, elapsed = self.predict_one(s["membership"])
            total_time += elapsed
            y_true.append(s["label"])
            y_pred.append(y_hat)
            probabilities.append(proba)
        positive_class = self.class_tokens[-1]
        y_score = [proba.get(positive_class, 0.0) for proba in probabilities]
        result = classification_metrics(y_true, y_pred, y_score, self.class_tokens)
        result.update({
            "total_time_s": total_time,
            "avg_time_per_query_ms": (total_time / len(samples)) * 1000,
            "y_true": y_true, "y_pred": y_pred,
            "y_score": y_score, "probabilities": probabilities,
        })
        return result


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples

    fkg = generate_synthetic_fkg(n_rules=150, seed=1)
    test_samples = generate_synthetic_test_samples(fkg, n_samples=100, seed=20)

    print("=== Node2Vec-lite + kNN ===")
    n2v = Node2VecLite(fkg, d=16, epochs=10, seed=1).fit()
    clf = KNNOnEmbedding(fkg, n2v.E, fkg.token2idx, k=5)
    res = clf.evaluate(test_samples)
    print(f"Accuracy={res['accuracy']:.4f}  F1={res['f1_macro']:.4f}  "
          f"(train_time={n2v.train_time_s:.2f}s)")
    assert res["accuracy"] > 0.3, "Node2Vec+kNN quá tệ -- kiểm tra lại!"

    print("=== TransE-lite + kNN ===")
    transe = TransELite(fkg, d=16, epochs=10, seed=1).fit()
    clf = KNNOnEmbedding(fkg, transe.E, fkg.token2idx, k=5)
    res = clf.evaluate(test_samples)
    print(f"Accuracy={res['accuracy']:.4f}  F1={res['f1_macro']:.4f}  "
          f"(train_time={transe.train_time_s:.2f}s)")

    print("=== DistMult-lite + kNN ===")
    dm = DistMultLite(fkg, d=16, epochs=10, seed=1).fit()
    clf = KNNOnEmbedding(fkg, dm.E, fkg.token2idx, k=5)
    res = clf.evaluate(test_samples)
    print(f"Accuracy={res['accuracy']:.4f}  F1={res['f1_macro']:.4f}  "
          f"(train_time={dm.train_time_s:.2f}s)")
    print("\nKiểm thử 3 baseline: hoàn tất, không lỗi runtime.")
