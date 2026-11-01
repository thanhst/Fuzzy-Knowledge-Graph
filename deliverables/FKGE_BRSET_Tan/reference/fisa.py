"""
models/fisa.py — Suy diễn FISA theo đúng mô hình FKG-Pairs như cài đặt FKGS v3.2
(src/fisa_pairs.py) và Chương 2–3 luận án:

  (7)  A^t_{c'} = |{s : s khớp t trên bộ k+1 thuộc tính c'}| / |R|
  (8)  B^t_c    = (Σ_{c'} A^t_{c'}) · min_{i ∈ c} M^t_i ,  M^t_i = |{s : s_i = t_i, nhãn(s) = nhãn(t)}| / |R|
  (3.Wfisa) W_{c,v,l} = Σ_{t : nhãn(t) = l, v_t(c) = v} B^t_c         (gộp CÓ ĐIỀU KIỆN THEO NHÃN)
  suy diễn mờ (Định nghĩa 3.fisamm, Chương 3):
       W̃_{c,l}(x) = Σ_v (Π_{j ∈ c} μ_{v_j}(x_j)) · W_{c,v,l} = Σ_{t: nhãn(t)=l} m_{t,c}(x) B^t_c
       D_l(x)     = max_c W̃_{c,l}(x) + min_c W̃_{c,l}(x)
  quyết định: "argmax" (công thức (12) của FKG-Pairs) | "ratio9" (mã notebook gốc: D0 > 9·D1)
              | "calibrated" (luận án: ngưỡng trên s = log D1 − log D0 chọn trên tập xác thực).

Khi độ thuộc là véc-tơ rời rạc (một nhãn ngôn ngữ có độ thuộc 1), suy diễn mờ trùng với FISA
rời rạc của FKGS v3.2. Nút gồm k = config.FISA.NODE_SIZE thuộc tính (mặc định 3 như notebook gốc).

Hai cách cài đặt cùng một hàm quyết định:
  FISA            : bảng W tính sẵn theo nút; mỗi truy vấn tra nhiều nhất 2^k bộ mức trên mỗi nút.
  FISASequential  : "FISA gốc" — duyệt tuần tự mọi luật trên mọi nút, chi phí O(|R|·C(r,k)) mỗi truy vấn;
                    dùng làm mốc chi phí T_F cho ràng buộc (R4) của Bài toán 3.1A.

Yêu cầu dữ liệu: mỗi luật phải có đủ một nhãn ngôn ngữ cho MỌI thuộc tính (luật sinh từ mẫu, như
FKG-MM/FKGS). Luật thiếu thuộc tính chỉ tham gia các bộ/nút mà nó có đủ thuộc tính (có cảnh báo).
"""
import time
import math
import warnings
from itertools import combinations, product
from collections import Counter
import numpy as np

try:
    import config as _C
    _CFG = getattr(_C, "FISA", None)
except Exception:  # pragma: no cover
    _CFG = None

DECISIONS = ("argmax", "ratio9", "calibrated")
EPS0 = 1e-12


def _cfg(name, default):
    return getattr(_CFG, name, default) if _CFG is not None else default


def _split(token):
    a, lv = token.rsplit("-", 1)
    return a, lv


class FISA:
    def __init__(self, fkg, node_size=None, decision=None, temperature=None, aggregator=None, eps=None):
        self.fkg = fkg
        self.aggregator = aggregator or _cfg("AGGREGATOR", "geomean")
        self.eps = float(eps if eps is not None else _cfg("EPS", 1e-9))
        self.class_tokens = list(fkg.class_tokens)
        self.k_req = int(node_size or _cfg("NODE_SIZE", 3))
        self.decision = decision or _cfg("DECISION", "calibrated")
        if self.decision not in DECISIONS:
            raise ValueError(f"decision phải thuộc {DECISIONS}")
        self.temperature = float(temperature or _cfg("TEMPERATURE", 1.0))
        self.threshold = None
        self.fit_time = 0.0

    # ------------------------------------------------------------ xây đồ thị
    def fit(self, val_samples=None):
        t0 = time.time()
        attrs = sorted({_split(t)[0] for t in self.fkg.vocab if not t.startswith("class-")})
        self.attrs = attrs
        self.attr_idx = {a: i for i, a in enumerate(attrs)}
        levels = {a: sorted({_split(t)[1] for t in self.fkg.vocab
                             if not t.startswith("class-") and _split(t)[0] == a}) for a in attrs}
        self.level_idx = {a: {lv: i for i, lv in enumerate(levels[a])} for a in attrs}
        self.n_levels = np.array([max(1, len(levels[a])) for a in attrs], dtype=np.int64)
        n, m = len(self.fkg.rules), len(attrs)
        if n == 0:
            raise ValueError("FISA: tập luật rỗng")
        codes = np.full((n, m), -1, dtype=np.int64)
        for r, rule in enumerate(self.fkg.rules):
            for tok in rule["antecedent_tokens"]:
                a, lv = _split(tok)
                codes[r, self.attr_idx[a]] = self.level_idx[a][lv]
        if (codes < 0).any():
            warnings.warn("FISA (FKG-Pairs): có luật thiếu thuộc tính; luật đó chỉ tham gia các bộ/nút "
                          "mà nó có đủ thuộc tính. Mô hình gốc giả thiết luật có đủ mọi thuộc tính.")
        self.codes = codes
        self.lab = np.array([self.class_tokens.index(r["consequent_token"]) for r in self.fkg.rules])
        self.rule_ids = [r.get("id", i) for i, r in enumerate(self.fkg.rules)]
        self.k = min(self.k_req, m)
        k_a = min(self.k + 1, m)
        nL = len(self.class_tokens)

        def radix(idx):
            key = np.zeros(n, dtype=np.int64)
            for j in idx:
                key = key * self.n_levels[j] + codes[:, j]
            return key

        # (7) tổng A^t trên các bộ k+1 thuộc tính mà luật t có đủ
        sumA = np.zeros(n)
        for c in combinations(range(m), k_a):
            valid = np.all(codes[:, list(c)] >= 0, axis=1)
            if not valid.any():
                continue
            key = radix(c)[valid]
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            sumA[valid] += cnt[inv] / n
        # M^t_i: cùng mức ngôn ngữ ở thuộc tính i và cùng nhãn với luật t
        M = np.zeros((n, m))
        for i in range(m):
            valid = codes[:, i] >= 0
            key = codes[valid, i] * nL + self.lab[valid]
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            M[valid, i] = cnt[inv] / n
        self.sumA, self.M = sumA, M
        # (8) B^t_c và bảng (3.Wfisa) dạng mảng dày: Wd[p, bộ mức, nhãn], bộ mức mã hoá cơ số Lmax
        self.nodes = list(combinations(range(m), self.k))
        self.Lmax = int(self.n_levels.max())
        nV = self.Lmax ** self.k
        self.B = np.zeros((n, len(self.nodes)))
        self.Wd = np.zeros((len(self.nodes), nV, nL))
        for p, c in enumerate(self.nodes):
            valid = np.all(codes[:, list(c)] >= 0, axis=1)
            Bc = np.where(valid, sumA * M[:, list(c)].min(axis=1), 0.0)
            self.B[:, p] = Bc
            if valid.any():
                key = np.zeros(n, dtype=np.int64)
                for j in c:
                    key = key * self.Lmax + np.maximum(codes[:, j], 0)
                np.add.at(self.Wd[p], (key[valid], self.lab[valid]), Bc[valid])
        self.Wnz = self.Wd.sum(axis=2) > 0          # bộ mức có ít nhất một luật đóng góp dương
        self.node_idx = np.array(self.nodes, dtype=np.int64).reshape(len(self.nodes), self.k)
        cnt = Counter(self.lab.tolist())
        self.fallback = self.class_tokens[max(cnt, key=cnt.get)]
        self.fit_time = time.time() - t0
        if self.decision == "calibrated" and val_samples and len(self.class_tokens) == 2:
            self.calibrate(val_samples)
        return self

    # ------------------------------------------------------------ suy diễn
    def _mu(self, membership):
        """Độ thuộc theo thuộc tính: list[(mã mức, μ)] với μ > 0; thuộc tính không quan sát -> []."""
        mu = [[] for _ in self.attrs]
        for tok, deg in membership.items():
            if deg <= 0 or tok.startswith("class-"):
                continue
            a, lv = _split(tok)
            if a in self.attr_idx and lv in self.level_idx[a]:
                mu[self.attr_idx[a]].append((self.level_idx[a][lv], float(deg)))
        return mu

    def _node_values(self, mu):
        """W̃_{c,l}(x) cho mọi nút (mảng số nút x số nhãn) — tra bảng tính sẵn: với mỗi nút, tích ngoài
        các véc-tơ độ thuộc của k thuộc tính (nhiều nhất 2^k thành phần khác 0) nhân với bảng W."""
        U = np.zeros((len(self.attrs), self.Lmax))
        for j, lst in enumerate(mu):
            for code, deg in lst:
                U[j, code] = deg
        P = U[self.node_idx[:, 0]]
        for q in range(1, self.k):
            P = (P[:, :, None] * U[self.node_idx[:, q]][:, None, :]).reshape(len(self.nodes), -1)
        V = np.einsum("pv,pvl->pl", P, self.Wd)
        any_match = bool(((P > 0) & self.Wnz).any())
        return V, any_match

    def decision_values(self, membership):
        """(D: dict nhãn -> D_l, ma trận W̃ theo nút, có khớp nút nào hay không)."""
        V, any_match = self._node_values(self._mu(membership))
        if self.aggregator == "geomean":       # FISA trung bình nhân: mọi nút đều góp bằng chứng
            D = np.exp(np.log(V + self.eps).mean(axis=0))
        else:                                  # FISA gốc: chỉ nút lớn nhất và nhỏ nhất
            D = V.max(axis=0) + V.min(axis=0)
        return {c: float(D[i]) for i, c in enumerate(self.class_tokens)}, V, any_match

    def _score(self, D):
        c0, c1 = self.class_tokens[0], self.class_tokens[1]
        return math.log(D[c1] + EPS0) - math.log(D[c0] + EPS0)

    def _decide(self, D, matched, decision=None):
        decision = decision or self.decision
        if not matched:
            return self.fallback
        if len(self.class_tokens) == 2:
            if decision == "calibrated" and self.threshold is not None:
                return self.class_tokens[1] if self._score(D) > self.threshold else self.class_tokens[0]
            if decision == "ratio9":
                c0, c1 = self.class_tokens
                return c0 if D[c0] > 9 * D[c1] else c1
        return max(self.class_tokens, key=lambda c: (D[c], -self.class_tokens.index(c)))

    def predict_one(self, membership):
        """Trả về (D_l dict, nhãn dự đoán, thời gian suy diễn)."""
        t0 = time.time()
        D, _, matched = self.decision_values(membership)
        y_hat = self._decide(D, matched)
        return D, y_hat, time.time() - t0

    def predict_proba_one(self, membership, temperature=None):
        """Phân phối FISA hiệu chỉnh p_F^cal (Định nghĩa 3.18): softmax((log(D_l+ε0) + β_l)/T), với
        β theo ngưỡng hiệu chỉnh (β_1 − β_0 = −τ); argmax của nó trùng quy tắc quyết định."""
        t0 = time.time()
        D, _, matched = self.decision_values(membership)
        y_hat = self._decide(D, matched)
        n = len(self.class_tokens)
        if not matched:
            return {c: 1.0 / n for c in self.class_tokens}, y_hat, time.time() - t0
        T = max(float(temperature or self.temperature), 1e-6)
        beta = np.zeros(n)
        if n == 2 and self.decision == "calibrated" and self.threshold is not None:
            beta[1] = -self.threshold
        v = (np.log(np.array([D[c] for c in self.class_tokens]) + EPS0) + beta) / T
        e = np.exp(v - v.max())
        p = e / e.sum()
        return {c: float(x) for c, x in zip(self.class_tokens, p)}, y_hat, time.time() - t0

    def attribution(self, membership):
        """Véc-tơ đóng góp luật a_F (Định nghĩa 3.attrF): dict id luật -> đóng góp chuẩn hoá vào D của
        nhãn dự đoán, qua hai nút đạt min và max."""
        D, V, matched = self.decision_values(membership)
        y_hat = self._decide(D, matched)
        l = self.class_tokens.index(y_hat)
        R_l = np.where(self.lab == l)[0]
        if not matched or D[y_hat] <= 0:
            return {self.rule_ids[t]: 1.0 / len(R_l) for t in R_l}
        col = V[:, l]
        U = np.zeros((len(self.attrs), self.Lmax + 1))           # cột cuối = 0 cho mã -1 (thiếu thuộc tính)
        for j, lst in enumerate(self._mu(membership)):
            for code, deg in lst:
                U[j, code] = deg
        contrib = np.zeros(len(self.rule_ids))
        if self.aggregator == "geomean":
            # Căn cứ luật của FISA trung bình nhân: mỗi nút có luật khớp góp phần bằng nhau (1/số nút),
            # chia cho các luật khớp tại nút theo tỉ lệ B (phân rã của log D = trung bình theo nút).
            nodes = [p for p in range(len(self.nodes)) if col[p] > 0]
            for p in nodes:
                w = np.ones(len(R_l))
                for j in self.nodes[p]:
                    w *= U[j, self.codes[R_l, j]]
                w = w * self.B[R_l, p]
                if w.sum() > 0:
                    contrib[R_l] += w / w.sum() / len(nodes)
        else:
            pmin, pmax = int(np.argmin(col)), int(np.argmax(col))
            for p in (pmin, pmax):
                w = np.ones(len(R_l))
                for j in self.nodes[p]:
                    w *= U[j, self.codes[R_l, j]]
                contrib[R_l] += w * self.B[R_l, p]
        s = contrib[R_l].sum()
        if s <= 0:
            return {self.rule_ids[t]: 1.0 / len(R_l) for t in R_l}
        return {self.rule_ids[t]: float(contrib[t] / s) for t in R_l if contrib[t] > 0}

    # ------------------------------------------------------------ hiệu chỉnh và đánh giá
    def calibrate(self, val_samples, criterion=None):
        """Chọn ngưỡng τ trên s = log D1 − log D0 bằng tập xác thực (FKGS v3.1–v3.2)."""
        criterion = criterion or _cfg("CALIB_CRITERION", "bacc")
        s, y = [], []
        for smp in val_samples:
            D, _, matched = self.decision_values(smp["membership"])
            s.append(self._score(D) if matched else 0.0)
            y.append(1 if smp["label"] == self.class_tokens[1] else 0)
        s, y = np.array(s), np.array(y)
        if len(np.unique(y)) < 2:
            self.threshold = 0.0
            return self.threshold
        cand = np.unique(np.concatenate([[0.0], np.quantile(s, np.linspace(0.01, 0.99, 197))]))
        best, bt = -1.0, 0.0
        for t in cand:
            p = (s > t).astype(int)
            if criterion == "acc":
                v = float(np.mean(p == y))
            else:
                v = 0.5 * (np.mean(p[y == 1] == 1) + np.mean(p[y == 0] == 0))
            if v > best + 1e-12:
                best, bt = v, float(t)
        self.threshold = bt
        return bt

    def evaluate(self, samples):
        """Accuracy, F1 macro, độ chính xác cân bằng, AUC (hai lớp), accuracy theo argmax và theo
        quy tắc notebook (D0 > 9·D1), số truy vấn không khớp nút nào, thời gian."""
        y_true, y_pred, y_am, y_r9, scores = [], [], [], [], []
        total, unseen = 0.0, 0
        for s in samples:
            t0 = time.time()
            D, _, matched = self.decision_values(s["membership"])
            yh = self._decide(D, matched)
            total += time.time() - t0
            unseen += (not matched)
            y_true.append(s["label"]); y_pred.append(yh)
            y_am.append(self._decide(D, matched, "argmax"))
            y_r9.append(self._decide(D, matched, "ratio9"))
            if len(self.class_tokens) == 2:
                scores.append(self._score(D) if matched else 0.0)
        out = {
            "accuracy": float(np.mean([a == b for a, b in zip(y_true, y_pred)])),
            "f1_macro": _macro_f1(y_true, y_pred, self.class_tokens),
            "balanced_accuracy": _balanced_acc(y_true, y_pred, self.class_tokens),
            "accuracy_argmax": float(np.mean([a == b for a, b in zip(y_true, y_am)])),
            "balanced_accuracy_argmax": _balanced_acc(y_true, y_am, self.class_tokens),
            "accuracy_ratio9": float(np.mean([a == b for a, b in zip(y_true, y_r9)])),
            "n_unseen": unseen, "threshold": self.threshold, "decision": self.decision, "node_size": self.k,
            "total_time_s": total, "avg_time_per_query_ms": total / len(samples) * 1000,
            "fit_time_s": self.fit_time, "y_true": y_true, "y_pred": y_pred,
        }
        if len(self.class_tokens) == 2:
            out["auc"] = _auc([1 if y == self.class_tokens[1] else 0 for y in y_true], scores)
        return out


class FISASequential(FISA):
    """FISA gốc: duyệt tuần tự mọi luật trên mọi nút cho mỗi truy vấn (cùng hàm quyết định với FISA,
    chi phí O(|R|·C(r,k)) mỗi truy vấn) — mốc chi phí T_F của ràng buộc (R4)."""

    def _node_values(self, mu):
        nL = len(self.class_tokens)
        mud = [dict(x) for x in mu]
        out = np.zeros((len(self.nodes), nL))
        any_match = False
        lut = [np.array([mud[j].get(v, 0.0) for v in range(int(self.n_levels[j]))] + [0.0])
               for j in range(len(self.attrs))]
        for p, c in enumerate(self.nodes):
            w = np.ones(len(self.lab))
            for j in c:
                w *= lut[j][self.codes[:, j]]          # mã -1 -> phần tử cuối = 0
            contrib = w * self.B[:, p]
            if contrib.any():
                any_match = True
            out[p] = np.bincount(self.lab, weights=contrib, minlength=nL)
        return out, any_match


# ---------------------------------------------------------------- tiện ích đánh giá
def majority_class_baseline(samples):
    labels = [s["label"] for s in samples]
    top_label, top_count = Counter(labels).most_common(1)[0]
    return top_count / len(samples), top_label


def warn_if_not_beating_majority(result, samples, model_name="Mô hình", margin=0.02):
    """Cảnh báo nếu mô hình không vượt rõ baseline lớp đa số (dấu hiệu sụp về lớp đa số)."""
    base_acc, base_label = majority_class_baseline(samples)
    acc = result["accuracy"]
    if acc <= base_acc + margin:
        print(f"  !! CẢNH BÁO: {model_name} có accuracy={acc:.4f}, KHÔNG vượt rõ rệt baseline lớp đa số "
              f"({base_acc:.4f}, luôn đoán '{base_label}'). Kiểm tra độ chính xác cân bằng và ngưỡng quyết định.")
        return False
    return True


def _macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _balanced_acc(y_true, y_pred, classes):
    recs = []
    for c in classes:
        n_c = sum(1 for a in y_true if a == c)
        if n_c:
            recs.append(sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c) / n_c)
    return float(np.mean(recs)) if recs else 0.0


def _auc(y, s):
    """AUC theo thống kê Mann–Whitney với hạng trung bình cho giá trị bằng nhau."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); sv = s[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples
    fkg = generate_synthetic_fkg(n_rules=300, seed=1)
    samples = generate_synthetic_test_samples(fkg, n_samples=300, seed=2)
    f = FISA(fkg).fit(val_samples=samples[:150])
    res = f.evaluate(samples[150:])
    print(f"FISA (FKG-Pairs k={f.k}): acc={res['accuracy']:.4f} bacc={res['balanced_accuracy']:.4f} "
          f"AUC={res.get('auc', float('nan')):.4f} ({res['avg_time_per_query_ms']:.3f} ms/truy vấn)")
    fs = FISASequential(fkg).fit(val_samples=samples[:150])
    for s in samples[150:170]:
        a, _, _ = f.decision_values(s["membership"]); b, _, _ = fs.decision_values(s["membership"])
        assert all(abs(a[c] - b[c]) < 1e-9 for c in a), "FISA và FISASequential phải cho cùng D"
    print("Kiểm thử FISA: OK (bảng tra và tuần tự trùng nhau)")
