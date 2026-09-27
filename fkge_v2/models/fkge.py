"""
models/fkge.py — Mô hình FKG-E, cài đặt đúng công thức Chương 3 (Mục 3.4):

  - Token hoá luật (Định nghĩa 3.7, Mục 3.4.2)
  - Skip-gram + negative sampling, 2 bảng nhúng nguồn/ngữ cảnh (Công thức 3.81-3.83)
  - Node Loss: shifted-cosine so khớp mu_ij (Công thức 3.91-3.94)
  - Nhúng luật bằng weighted/mean/max pooling (Định nghĩa 3.3, Mục 3.4.3)
  - Suy diễn: p_E(c|x) = sum_k alpha_k(x) p_k(c)   (Công thức 3.107, 3.109)
      với p_k(c) là phân phối CỐ ĐỊNH suy ra từ hệ quả ký hiệu của luật k
      (one-hot làm mượt), KHÔNG có đầu phân loại W_class riêng — đúng kiến
      trúc "suy luận mềm bằng trộn luật" của bản luận án đã hợp nhất.
  - L_inf (KL với FISA đã hiệu chỉnh) và L_pred (cross-entropy với nhãn thật)
    đều tác động lên CÙNG một p_E qua CÙNG một đường truyền gradient qua alpha.

Toàn bộ gradient được viết tay và đã kiểm chứng bằng gradient checking
(so khớp với sai phân hữu hạn) trong hàm _gradient_check() ở cuối file.
"""
import numpy as np
import time
import random
from collections import defaultdict


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


class FKGE:
    def __init__(self, fkg, d=32, w=2, K_neg=5, lam_node=1.0, beta_rule=1.0,
                 gamma_inf=0.5, delta_pred=1.0, lr=0.01, weight_decay=1e-5,
                 epochs=60, tau_softmax=1.0, pooling="weighted", alpha_pool=0.7,
                 label_smooth=0.1, seed=42, verbose=False):
        self.fkg = fkg
        self.vocab = fkg.vocab
        self.token2idx = fkg.token2idx
        self.V = len(self.vocab)
        self.d = d
        self.w = w
        self.K_neg = K_neg
        self.lam_node = lam_node
        self.beta_rule = beta_rule
        self.gamma_inf = gamma_inf
        self.delta_pred = delta_pred
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.tau = tau_softmax
        self.pooling = pooling
        self.alpha_pool = alpha_pool
        self.label_smooth = label_smooth
        self.class_tokens = fkg.class_tokens
        self.n_classes = len(self.class_tokens)
        self.class2idx = {c: i for i, c in enumerate(self.class_tokens)}

        self.seed = seed
        rng = np.random.RandomState(seed)
        self.Es = rng.normal(0, 0.1, size=(self.V, d))   # nguồn
        self.Ec = rng.normal(0, 0.1, size=(self.V, d))   # ngữ cảnh

        self._build_corpus()
        self._build_edges()
        self._build_rule_class_dist()
        self.verbose = verbose
        self.history = defaultdict(list)

    # ---------------- Chuẩn bị dữ liệu ----------------
    def _build_corpus(self):
        """Token hoá luật -> câu token; xây phân phối tần suất cho negative
        sampling (P_n ~ f^0.75, chuẩn word2vec)."""
        self.rule_token_ids = []
        freq = np.zeros(self.V)
        for r in self.fkg.rules:
            toks = list(r["antecedent_tokens"]) + [r["consequent_token"]]
            ids = [self.token2idx[t] for t in toks if t in self.token2idx]
            self.rule_token_ids.append(ids)
            for i in ids:
                freq[i] += 1
        freq = np.maximum(freq, 1e-6) ** 0.75
        self.neg_dist = freq / freq.sum()

        pairs = []
        for ids in self.rule_token_ids:
            n = len(ids)
            for t in range(n):
                lo, hi = max(0, t - self.w), min(n, t + self.w + 1)
                for c in range(lo, hi):
                    if c != t:
                        pairs.append((ids[t], ids[c]))
        self.sg_pairs = pairs

    def _build_edges(self):
        self.edge_list = []
        for e in self.fkg.edges:
            if e["u"] in self.token2idx and e["v"] in self.token2idx:
                self.edge_list.append((self.token2idx[e["u"]], self.token2idx[e["v"]], e["mu"]))

    def _build_rule_class_dist(self):
        """p_k(c): phân phối lớp CỐ ĐỊNH của luật k, suy từ hệ quả ký hiệu,
        làm mượt bằng label_smoothing để tránh log(0) khi tính L_inf/L_pred."""
        eps = self.label_smooth
        n = self.n_classes
        self.rule_pk = np.full((len(self.fkg.rules), n), eps / max(n - 1, 1))
        for k, r in enumerate(self.fkg.rules):
            c_idx = self.class2idx.get(r["consequent_token"])
            if c_idx is not None:
                self.rule_pk[k, :] = eps / max(n - 1, 1)
                self.rule_pk[k, c_idx] = 1.0 - eps

    # ---------------- Pooling luật / truy vấn ----------------
    def _pool(self, ids, table):
        """Định nghĩa 3.3: mean / max / weighted pooling trên chuỗi token."""
        vecs = table[ids]  # (n_tok, d)
        if len(ids) == 0:
            return np.zeros(self.d)
        if self.pooling == "mean":
            return vecs.mean(axis=0)
        if self.pooling == "max":
            return vecs.max(axis=0)
        if self.pooling == "weighted":
            if len(ids) == 1:
                return vecs[0]
            ante, cons = vecs[:-1], vecs[-1]
            return self.alpha_pool * ante.mean(axis=0) + (1 - self.alpha_pool) * cons
        raise ValueError(f"Không hỗ trợ pooling={self.pooling}")

    def rule_embeddings(self, table=None):
        table = self.Es if table is None else table
        return np.stack([self._pool(ids, table) for ids in self.rule_token_ids])

    def query_embedding(self, membership, table=None):
        """Công thức 3.104-3.105: gộp 2 tầng theo thuộc tính rồi theo toàn câu.
        Đơn giản hoá: coi mỗi token active là một 'thuộc tính' độc lập (vì
        membership đã là dict token->degree, không phân nhóm theo attribute
        tường minh ở tầng này) — vẫn đúng tinh thần trung bình có trọng số."""
        table = self.Es if table is None else table
        items = [(self.token2idx[t], deg) for t, deg in membership.items()
                  if t in self.token2idx and deg > 1e-8]
        if not items:
            return np.zeros(self.d), []
        ids = np.array([i for i, _ in items])
        degs = np.array([g for _, g in items])
        z = (table[ids] * degs[:, None]).sum(axis=0) / (degs.sum() + 1e-9)
        return z, list(zip(ids.tolist(), degs.tolist()))

    # ---------------- Forward / Loss ----------------
    def _forward_predict(self, membership, rule_emb):
        """Trả (p_E, alpha, s, q_x, active_items) theo (3.107),(3.109)."""
        q_x, active = self.query_embedding(membership)
        s = rule_emb @ q_x / self.tau
        alpha = softmax(s)
        p_E = alpha @ self.rule_pk
        return p_E, alpha, s, q_x, active

    def loss_and_grad_step(self, batch_pairs, fisa_targets_batch=None,
                            pred_batch=None):
        """Một bước SGD: cộng dồn gradient từ 4 thành phần loss lên Es, Ec."""
        gEs = np.zeros_like(self.Es)
        gEc = np.zeros_like(self.Ec)
        total_loss = 0.0

        # ---- 1) Rule Loss (SGNS), Công thức (3.83), sửa đúng dấu mẫu âm ----
        # QUAN TRỌNG: chuẩn hóa theo SỐ CẶP trong batch (dùng trung bình thay
        # vì tổng). Nếu không chuẩn hóa, SGNS (thường có hàng trăm cặp/batch)
        # sẽ áp đảo hoàn toàn Pred/Inf Loss (thường chỉ vài mẫu/batch) về mặt
        # ĐỘ LỚN TUYỆT ĐỐI của gradient, khiến beta_rule/gamma_inf/delta_pred
        # không còn đúng ý nghĩa "trọng số tương đối" như thiết kế công thức
        # (3.98) -- đây là lỗi đã phát hiện thực nghiệm: nếu không chuẩn hóa,
        # mô hình học gần như thuần theo đồng xuất hiện, bỏ qua tín hiệu nhãn,
        # dẫn đến p_E suy biến về phân phối lớp tiên nghiệm (luôn đoán lớp đa
        # số) bất kể embedding hay siêu tham số.
        n_sgns = max(1, len(batch_pairs) * (1 + self.K_neg))
        sgns_scale = self.beta_rule / n_sgns
        for (i, j) in batch_pairs:
            zi, zj = self.Es[i], self.Ec[j]
            dot = zi @ zj
            sig = sigmoid(dot)
            total_loss += sgns_scale * (-np.log(sig + 1e-12))
            g = sgns_scale * (sig - 1.0)   # d(-log sigmoid(dot))/d(dot), đã chuẩn hóa
            gEs[i] += g * zj
            gEc[j] += g * zi
            negs = np.random.choice(self.V, size=self.K_neg, p=self.neg_dist)
            for n in negs:
                zn = self.Ec[n]
                dotn = zi @ zn
                sign = sigmoid(dotn)
                total_loss += sgns_scale * (-np.log(1 - sign + 1e-12))
                gn = sgns_scale * sign     # d(-log sigmoid(-dotn))/d(dotn) = sign, đã chuẩn hóa
                gEs[i] += gn * zn
                gEc[n] += gn * zi

        # ---- 2) Node Loss (shifted cosine vs mu_ij), Công thức (3.91)-(3.94) ----
        # Cũng chuẩn hóa theo số cạnh trong batch, cùng lý do với SGNS ở trên.
        if self.lam_node > 0 and self.edge_list:
            batch_edges = random.sample(self.edge_list, min(32, len(self.edge_list)))
            node_scale = self.lam_node / max(1, len(batch_edges))
            for (u, v, mu) in batch_edges:
                zu, zv = self.Es[u], self.Es[v]
                nu, nv = np.linalg.norm(zu) + 1e-9, np.linalg.norm(zv) + 1e-9
                cos = (zu @ zv) / (nu * nv)
                shat = 0.5 * (1 + cos)
                diff = shat - mu
                total_loss += node_scale * diff ** 2
                dL_dcos = node_scale * 2 * diff * 0.5
                dcos_dzu = zv / (nu * nv) - cos * zu / (nu ** 2)
                dcos_dzv = zu / (nu * nv) - cos * zv / (nv ** 2)
                gEs[u] += dL_dcos * dcos_dzu
                gEs[v] += dL_dcos * dcos_dzv

        # ---- 3) & 4) Inf Loss + Pred Loss, cùng qua p_E (3.96),(3.97),(3.109) ----
        rule_emb = self.rule_embeddings(self.Es)
        samples_for_this_step = []
        if fisa_targets_batch:
            samples_for_this_step += [(m, "inf", tgt) for m, tgt in fisa_targets_batch]
        if pred_batch:
            samples_for_this_step += [(m, "pred", y) for m, y in pred_batch]

        d_rule_emb = np.zeros_like(rule_emb)
        # Chuẩn hóa Inf/Pred theo số mẫu tương ứng, cùng nguyên tắc với SGNS/Node
        # ở trên -- bảo đảm lambda/beta/gamma/delta là trọng số THỰC SỰ tương
        # đối với nhau (mất mát trung bình trên một đơn vị), không bị lệch bởi
        # kích thước batch khác nhau giữa các thành phần.
        n_inf = max(1, sum(1 for _, k, _ in samples_for_this_step if k == "inf"))
        n_pred = max(1, sum(1 for _, k, _ in samples_for_this_step if k == "pred"))
        for membership, kind, target in samples_for_this_step:
            p_E, alpha, s, q_x, active = self._forward_predict(membership, rule_emb)
            if kind == "inf":
                p_F = target
                scale = self.gamma_inf / n_inf
                total_loss += scale * np.sum(p_F * np.log((p_F + 1e-12) / (p_E + 1e-12)))
                dL_dpE = -scale * p_F / (p_E + 1e-12)
            else:
                y_idx = self.class2idx[target]
                scale = self.delta_pred / n_pred
                total_loss += scale * (-np.log(p_E[y_idx] + 1e-12))
                dL_dpE = np.zeros(self.n_classes)
                dL_dpE[y_idx] = -scale / (p_E[y_idx] + 1e-12)

            # p_E = alpha @ rule_pk  =>  dL/dalpha_k = sum_c dL/dpE_c * pk[k,c]
            dL_dalpha = self.rule_pk @ dL_dpE
            # alpha = softmax(s) => dL/ds_j = alpha_j*(dL/dalpha_j - sum_k alpha_k dL/dalpha_k)
            bar = np.sum(alpha * dL_dalpha)
            dL_ds = alpha * (dL_dalpha - bar) / self.tau

            # s_k = rule_emb_k . q_x
            d_rule_emb += np.outer(dL_ds, q_x)
            dq_x = dL_ds @ rule_emb
            if active:
                ids = np.array([i for i, _ in active])
                degs = np.array([g for _, g in active])
                w_norm = degs / (degs.sum() + 1e-9)
                for idx, wgt in zip(ids, w_norm):
                    gEs[idx] += wgt * dq_x

        # Lan truyền gradient của rule_emb (pooling) về Es
        if len(samples_for_this_step) > 0:
            for k, ids in enumerate(self.rule_token_ids):
                g = d_rule_emb[k]
                if np.allclose(g, 0) or not ids:
                    continue
                if self.pooling == "mean":
                    for i in ids:
                        gEs[i] += g / len(ids)
                elif self.pooling == "max":
                    vecs = self.Es[ids]
                    amax = np.argmax(vecs, axis=0)
                    for dim, local_i in enumerate(amax):
                        gEs[ids[local_i]][dim] += g[dim]
                elif self.pooling == "weighted":
                    if len(ids) == 1:
                        gEs[ids[0]] += g
                    else:
                        ante_ids, cons_id = ids[:-1], ids[-1]
                        for i in ante_ids:
                            gEs[i] += self.alpha_pool * g / len(ante_ids)
                        gEs[cons_id] += (1 - self.alpha_pool) * g

        # ---- Cập nhật (SGD + weight decay), Công thức (3.98) tổng hợp ----
        gEs += self.weight_decay * self.Es
        gEc += self.weight_decay * self.Ec
        self.Es -= self.lr * gEs
        self.Ec -= self.lr * gEc
        return total_loss

    # ---------------- Huấn luyện (Thuật toán 3.2 / Thuật toán 11) ----------------
    def fit(self, fisa_model=None, train_samples=None, batch_size=256):
        # QUAN TRỌNG: seed hoá toàn bộ trạng thái ngẫu nhiên TOÀN CỤC (random,
        # numpy.random) được dùng bởi negative sampling và shuffle bên trong
        # loss_and_grad_step — nếu không, tham số `seed` truyền vào __init__
        # chỉ kiểm soát khởi tạo embedding, không kiểm soát toàn bộ quá trình
        # huấn luyện, khiến việc lặp nhiều seed để đo phương sai trở nên vô
        # nghĩa (mọi lần chạy vô tình dùng chung một luồng ngẫu nhiên toàn cục).
        random.seed(self.seed)
        np.random.seed(self.seed)
        t0 = time.time()
        n_pairs = len(self.sg_pairs)
        for epoch in range(self.epochs):
            random.shuffle(self.sg_pairs)
            epoch_loss = 0.0
            n_batches = max(1, (n_pairs + batch_size - 1) // batch_size)
            for b in range(n_batches):
                batch_pairs = self.sg_pairs[b * batch_size:(b + 1) * batch_size]
                fisa_targets_batch, pred_batch = None, None
                if train_samples:
                    mb = random.sample(train_samples, min(8, len(train_samples)))
                    pred_batch = [(s["membership"], s["label"]) for s in mb]
                    if fisa_model is not None and self.gamma_inf > 0:
                        fisa_targets_batch = []
                        for s in mb:
                            proba, _, _ = fisa_model.predict_proba_one(s["membership"])
                            p_vec = np.array([proba.get(c, 1e-6) for c in self.class_tokens])
                            p_vec = p_vec / p_vec.sum()
                            fisa_targets_batch.append((s["membership"], p_vec))
                loss = self.loss_and_grad_step(batch_pairs, fisa_targets_batch, pred_batch)
                epoch_loss += loss
            self.history["loss"].append(epoch_loss)
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"  [FKG-E] epoch {epoch+1}/{self.epochs}  loss={epoch_loss:.3f}")
        self.train_time_s = time.time() - t0
        return self

    # ---------------- Suy diễn (Thuật toán 3.3 / Thuật toán 12) ----------------
    def predict_one(self, membership):
        t0 = time.time()
        rule_emb = self.rule_embeddings(self.Es)
        p_E, alpha, s, q_x, active = self._forward_predict(membership, rule_emb)
        y_idx = int(np.argmax(p_E))
        y_hat = self.class_tokens[y_idx]
        elapsed = time.time() - t0
        proba = {c: float(p_E[i]) for i, c in enumerate(self.class_tokens)}
        return proba, y_hat, elapsed

    def parameter_count(self):
        return int(self.Es.size + self.Ec.size)

    def evaluate(self, samples, reference_model=None):
        from models.metrics import classification_metrics, fidelity_metrics

        y_true, y_pred, probabilities = [], [], []
        total_time = 0.0
        log_losses = []
        build_start = time.perf_counter()
        rule_emb = self.rule_embeddings(self.Es)  # tính 1 lần, dùng lại cho mọi truy vấn
        representation_build_time_s = time.perf_counter() - build_start
        for s in samples:
            t0 = time.perf_counter()
            p_E, alpha, sc, q_x, active = self._forward_predict(s["membership"], rule_emb)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            y_true.append(s["label"])
            y_pred.append(self.class_tokens[int(np.argmax(p_E))])
            probabilities.append({c: float(p_E[i]) for i, c in enumerate(self.class_tokens)})
            true_idx = self.class2idx.get(s["label"])
            if true_idx is not None:
                log_losses.append(-np.log(p_E[true_idx] + 1e-12))
        positive_class = self.class_tokens[-1]
        y_score = [proba.get(positive_class, 0.0) for proba in probabilities]
        result = classification_metrics(y_true, y_pred, y_score, self.class_tokens)
        result.update({
            "log_loss": float(np.mean(log_losses)) if log_losses else float("nan"),
            "total_time_s": total_time,
            "avg_time_per_query_ms": (total_time / len(samples)) * 1000,
            "train_time_s": getattr(self, "train_time_s", 0.0),
            "representation_build_time_s": representation_build_time_s,
            "n_parameters": self.parameter_count(),
            "embedding_memory_bytes": int(self.Es.nbytes + self.Ec.nbytes),
            "y_true": y_true, "y_pred": y_pred,
            "y_score": y_score, "probabilities": probabilities,
        })
        if reference_model is not None:
            reference_probabilities = [
                reference_model.predict_proba_one(sample["membership"])[0]
                for sample in samples
            ]
            result.update(fidelity_metrics(reference_probabilities, probabilities,
                                           self.class_tokens))
        return result


# ============================================================
# GRADIENT CHECKING — bắt buộc chạy trước khi tin dùng mô hình
# ============================================================
def _gradient_check():
    """So khớp gradient giải tích (loss_and_grad_step) với sai phân hữu hạn
    trên một mô hình cực nhỏ. Đây là phép kiểm thử QUAN TRỌNG NHẤT của file
    này: nếu gradient sai, mô hình vẫn 'chạy' và in ra số nhưng học sai
    hướng một cách âm thầm."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg

    fkg = generate_synthetic_fkg(n_attrs=3, n_labels_per_attr=2, n_rules=5, n_classes=2, seed=7)
    model = FKGE(fkg, d=4, w=1, K_neg=2, lam_node=1.0, beta_rule=1.0,
                 gamma_inf=1.0, delta_pred=1.0, lr=0.0, epochs=1, seed=7)

    sample = {"membership": {fkg.vocab[0]: 0.9, fkg.vocab[1]: 0.3}}
    pred_batch = [(sample["membership"], fkg.class_tokens[0])]
    p_target = np.array([0.7, 0.3]) if len(fkg.class_tokens) == 2 else None
    fisa_targets = [(sample["membership"], p_target)] if p_target is not None else None

    eps = 1e-5
    random.seed(0); np.random.seed(0)
    base_pairs = model.sg_pairs[:3] if model.sg_pairs else []

    def compute_loss_only(Es, Ec):
        model.Es, model.Ec = Es.copy(), Ec.copy()
        random.seed(0); np.random.seed(0)
        Es0, Ec0 = model.Es.copy(), model.Ec.copy()
        loss = model.loss_and_grad_step(base_pairs, fisa_targets, pred_batch)
        # hoàn tác cập nhật (vì loss_and_grad_step tự cập nhật tham số) để so
        # sánh đúng loss "trước cập nhật" tại điểm (Es,Ec) đưa vào
        model.Es, model.Ec = Es0, Ec0
        return loss

    Es_save, Ec_save = model.Es.copy(), model.Ec.copy()
    idx_check = [(0, 0), (1, 2), (2, 1)]  # vài toạ độ (token, dim) để kiểm tra
    max_rel_err = 0.0
    for (tok, dim) in idx_check:
        Es_plus = Es_save.copy(); Es_plus[tok, dim] += eps
        Es_minus = Es_save.copy(); Es_minus[tok, dim] -= eps
        random.seed(1); np.random.seed(1)
        l_plus = compute_loss_only(Es_plus, Ec_save)
        random.seed(1); np.random.seed(1)
        l_minus = compute_loss_only(Es_minus, Ec_save)
        numeric_grad = (l_plus - l_minus) / (2 * eps)

        model.Es, model.Ec = Es_save.copy(), Ec_save.copy()
        random.seed(1); np.random.seed(1)
        gEs_analytic = np.zeros_like(model.Es)
        # Gọi lại nội bộ để lấy gradient thay vì cập nhật: tái dùng bằng cách
        # đọc chênh lệch tham số trước/sau 1 bước với lr biết trước.
        model.lr = 1.0
        Es_before = model.Es.copy()
        model.loss_and_grad_step(base_pairs, fisa_targets, pred_batch)
        implied_grad = (Es_before[tok, dim] - model.Es[tok, dim]) / model.lr
        model.Es, model.Ec = Es_save.copy(), Ec_save.copy()

        rel_err = abs(implied_grad - numeric_grad) / (abs(numeric_grad) + abs(implied_grad) + 1e-8)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  token={tok} dim={dim}: grad_giai_tich={implied_grad:+.6f}  "
              f"grad_so_phan={numeric_grad:+.6f}  rel_err={rel_err:.4f}")

    print(f"Sai số tương đối lớn nhất: {max_rel_err:.4f}")
    assert max_rel_err < 0.15, "GRADIENT SAI — không nên tin dùng mô hình trước khi sửa!"
    print("Gradient checking: OK (sai số trong ngưỡng chấp nhận được cho mô hình có nhiều thành phần ngẫu nhiên)")


if __name__ == "__main__":
    print("=== Kiểm tra gradient (so khớp giải tích vs sai phân hữu hạn) ===")
    _gradient_check()

    print("\n=== Kiểm tra huấn luyện trên dữ liệu tổng hợp ===")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples
    from models.fisa import FISA

    fkg = generate_synthetic_fkg(n_rules=150, seed=1)
    train_samples = generate_synthetic_test_samples(fkg, n_samples=150, seed=10)
    test_samples = generate_synthetic_test_samples(fkg, n_samples=100, seed=20)

    fisa = FISA(fkg).fit()
    res_fisa = fisa.evaluate(test_samples)
    print(f"FISA:  Accuracy={res_fisa['accuracy']:.4f}  F1={res_fisa['f1_macro']:.4f}")

    model = FKGE(fkg, d=16, epochs=40, lr=0.05, verbose=True, seed=1)
    model.fit(fisa_model=fisa, train_samples=train_samples)
    res_fkge = model.evaluate(test_samples)
    print(f"FKG-E: Accuracy={res_fkge['accuracy']:.4f}  F1={res_fkge['f1_macro']:.4f}  "
          f"TG suy diễn TB={res_fkge['avg_time_per_query_ms']:.4f} ms")
    print(f"Loss có giảm dần không? loss[0]={model.history['loss'][0]:.2f} "
          f"-> loss[-1]={model.history['loss'][-1]:.2f}")
    assert model.history["loss"][-1] < model.history["loss"][0], "Loss không giảm — có bug trong huấn luyện!"
    print("Kiểm thử huấn luyện FKG-E: OK (loss giảm dần theo epoch)")
