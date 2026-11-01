"""
models/fkge.py — Mô hình FKG-E cài đặt đúng Chương 3 (Mục 3.2.4 và Mục 3.3.2), tư tưởng word2vec trên hai
ma trận A, B của đồ thị tri thức mờ.

Ký hiệu (số hiệu theo bản LaTeX hiện hành):
  Nút giá trị 𝒱 = {(j, v)}: mỗi nút là một giá trị ngôn ngữ của một thuộc tính; L là tập nhãn.
  (3.5)  a_ii' = |{R : v_j(R) = v, v_k(R) = v'}| / |R|   (i = (j,v), i' = (k,v'), j ≠ k)  — trọng số cạnh A
  (3.6)  b_il  = |{R : v_j(R) = v, out(R) = l}| / |{R : v_j(R) = v}|                     — hàng của B
         a_i = Σ_i' a_ii',  |a| = Σ_i a_i,  P_n(n) = a_n/|a|,  b̃ = (1 − ε_b) b + ε_b/n_L.
  Θ = {z_i, c_i}_{i∈𝒱} ∪ {u_l}_{l∈L}.
  L_edge (bảo toàn cạnh, SGNS trên A) = −(1/|a|) Σ_{i≠i'} a_ii' [log σ(z_iᵀc_i') + K E_{n~P_n} log σ(−z_iᵀc_n)].
  L_node (bảo toàn nút, B)            = −(1/|𝒱_R|) Σ_{i∈𝒱_R} Σ_l b̃_il log softmax_l(z_iᵀu_l).
  Nhúng luật (3.pool): r_k = (α/r) Σ_j z_{(j, v_j(R_k))} + (1 − α) u_{out(R_k)}.
  Nhúng truy vấn (3.query): q(x) = (1/r) Σ_j Σ_v μ_{j,v}(x) z_{(j,v)}.
  s_k = cos_ε0(q, r_k);  s̄_l = τ_c log Σ_{k∈R_l} exp(s_k/τ_c);  p_E = (1−ε) softmax(s̄/τ) + ε/n_L.
  a_E = w_{ŷ,·}, w_{l,k} = softmax_{k∈R_l}(s_k/τ_c).
  Đích chưng cất từ FISA (Định nghĩa Giáo viên FISA): ŷ_F, e^F = a_F, p_F^cal = softmax((log(D+ε0)+β)/T).
  L_rule = mean KL(e^F ‖ w_{ŷ_F}),  L_inf = mean KL(p_F^cal ‖ p_E),  L_pred = −mean log p_E(y).
  Hàm mục tiêu (3.15):  L = λ_E L_edge + λ_N L_node + λ_R L_rule + λ_I L_inf + λ_P L_pred + λ_C ‖Θ‖².

Thuật toán huấn luyện (Thuật toán 4.1 luận án; optimizer = "gd_full", mặc định và là cấu hình của MỌI kết quả báo cáo):
GIẢM GRADIENT TOÀN PHẦN CÓ CHIẾU, tất định — mỗi vòng dùng ∇L trên toàn bộ cạnh E⁺, toàn bộ nút và toàn bộ mẫu huấn
luyện; kỳ vọng theo mẫu âm E_{n~P_n} tính CHÍNH XÁC (không lấy mẫu); Θ⁺ = Π_B(Θ − η∇L(Θ)) với Π_B là phép chiếu lên
hình cầu bán kính B_0 = √(L(Θ_0)/λ_C); bước η chọn bằng tìm kiếm theo đường quay lui (bắt đầu từ min(2η_trước, η_gd),
chia đôi) tới khi thoả điều kiện giảm đủ L(Θ⁺) ≤ L(Θ) − (η/2)‖G_η(Θ)‖², G_η = (Θ − Θ⁺)/η là ánh xạ gradient (trùng
∇L khi phép chiếu không kích hoạt). Dừng khi ‖∇L‖ ≤ ε_g, hoặc dừng sớm theo L_pred trên tập xác thực (patience), hoặc
hết T_ep vòng; trả về Θ có L_pred xác thực nhỏ nhất; chọn ngưỡng τ_E trên tập xác thực. Các tham số b, b_p, η chỉ dùng
cho biến thể theo lô "sgd"/"amsgrad" (lô cạnh lấy đều từ E⁺ với hệ số |E⁺|/b_p), không dùng trong kết quả báo cáo.
history ghi: grad_norm, loss, val_pred, eta, proj (phép chiếu có kích hoạt), ls_fail (tìm kiếm theo đường không đạt
điều kiện sau 40 lần chia đôi).
Khi cơ sở luật sinh từ chính các mẫu huấn luyện (FRB), luật cùng nhóm với mẫu bị loại khỏi suy diễn cho mẫu
đó (chống rò rỉ nhãn). Gradient tính bằng autograd.
"""
import time
import math
import warnings
from collections import defaultdict

import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp

try:
    import config as _C
    _CFG = getattr(_C, "FKGE", None)
except Exception:  # pragma: no cover
    _CFG = None


def _cfg(name, default):
    return getattr(_CFG, name, default) if _CFG is not None else default


# Tên tham số cũ (bản v2) -> tên theo Chương 3, để các kịch bản cũ chạy được.
_LEGACY = {"d": "d_e", "K_neg": "K", "lam_node": "lam_N", "beta_rule": "lam_E", "lam_S": "lam_E", "gamma_inf": "lam_I",
           "delta_pred": "lam_P", "lr": "eta", "epochs": "T_ep", "alpha_pool": "alpha",
           "tau_softmax": "tau", "batch_size": "b"}


def _log_sigmoid(x):
    return -anp.logaddexp(0.0, -x)


def _sigmoid(x):
    return 0.5 * (anp.tanh(0.5 * x) + 1.0)


def _cos(Q, R, eps0):
    num = anp.dot(Q, R.T)
    nq = anp.sqrt(anp.sum(Q * Q, axis=1) + eps0)
    nr = anp.sqrt(anp.sum(R * R, axis=1) + eps0)
    return num / (nq[:, None] * nr[None, :])


def _cos_rows(A, B, eps0):
    return anp.sum(A * B, axis=1) / (anp.sqrt(anp.sum(A * A, axis=1) + eps0) * anp.sqrt(anp.sum(B * B, axis=1) + eps0))


class FKGE:
    def __init__(self, fkg, d_e=None, K=None, alpha=None, tau=None, tau_c=None, eps=None, eps0=None, eps_b=None,
                 lam_E=None, lam_N=None, lam_R=None, lam_I=None, lam_P=None, lam_C=None, eta=None, optimizer=None,
                 eps_g=None, b=None, b_p=None, T_ep=None, patience=None, sigma=None, val_fraction=None,
                 pooling=None, seed=None, verbose=False, **legacy):
        for k, v in legacy.items():
            if k in _LEGACY:
                setattr(self, "_legacy_" + _LEGACY[k], v)   # ánh xạ tên cũ -> tên Chương 3
            elif k in ("weight_decay", "lam_A", "lam_B", "cooc", "w"):
                warnings.warn(f"FKGE: tham số '{k}' không còn dùng trong mô hình theo Chương 3; bỏ qua.", DeprecationWarning)
            else:
                raise TypeError(f"FKGE: tham số không hỗ trợ '{k}'")
        g = lambda name, val, default: (getattr(self, "_legacy_" + name) if hasattr(self, "_legacy_" + name)
                                        else (val if val is not None else _cfg(name, default)))
        self.fkg = fkg
        self.d_e = int(g("d_e", d_e, 32)); self.K = int(g("K", K, 5))
        self.tau = float(g("tau", tau, 0.1)); self.tau_c = float(g("tau_c", tau_c, 0.1))
        self.eps = float(g("eps", eps, 0.05)); self.eps0 = float(g("eps0", eps0, 1e-6))
        self.eps_b = float(g("eps_b", eps_b, 0.05))
        self.lam = {n: float(g("lam_" + n, v, dv)) for n, v, dv in
                    [("E", lam_E, 1.0), ("N", lam_N, 1.0), ("R", lam_R, 1.0), ("I", lam_I, 1.0), ("P", lam_P, 1.0),
                     ("C", lam_C, 1e-3)]}
        if self.lam["C"] <= 0:
            raise ValueError("λ_C phải > 0 (Mệnh đề tính bức; Nhận xét 3.coer)")
        if not (0 < self.eps_b < 1):
            raise ValueError("ε_b phải thuộc (0, 1) (Mệnh đề 3.coer(iii))")
        self.eta = float(g("eta", eta, 0.01)); self.optimizer = g("optimizer", optimizer, "gd_full")
        self.eta_gd = float(_cfg("eta_gd", 1.0))     # bước khởi đầu của tìm kiếm theo đường (gd_full)
        self.eps_g = float(g("eps_g", eps_g, 1e-4)); self.b = int(g("b", b, 64)); self.b_p = int(g("b_p", b_p, 256))
        self.T_ep = int(g("T_ep", T_ep, 100)); self.patience = int(g("patience", patience, 10))
        self.sigma = float(g("sigma", sigma, 0.1)); self.val_fraction = float(g("val_fraction", val_fraction, 0.2))
        if pooling == "mean":
            self.alpha = None   # gộp trung bình đều: α = r/(r+1) (Nhận xét 3.space)
        else:
            self.alpha = float(g("alpha", alpha, 0.7))
        self.seed = int(seed if seed is not None else _cfg("seed", 42))
        self.verbose = verbose
        self.class_tokens = list(fkg.class_tokens)
        self.nL = len(self.class_tokens)
        self.history = defaultdict(list)
        self._prepare()
        rng = np.random.RandomState(self.seed)
        nV, d = self.nV, self.d_e
        self.params = [rng.normal(0, self.sigma, (nV, d)), rng.normal(0, self.sigma, (nV, d)),
                       rng.normal(0, self.sigma, (self.nL, d))]          # z (nút), c (ngữ cảnh), u (nhãn)
        self.threshold = None
        self.teacher = None
        self.train_time_s = 0.0

    # ======================================================================== chuẩn bị (hằng số)
    def _prepare(self):
        fkg = self.fkg
        self.vocab = list(fkg.vocab)
        self.nodes = [t for t in self.vocab if not t.startswith("class-")]      # nút giá trị 𝒱
        self.node_idx = {t: i for i, t in enumerate(self.nodes)}
        self.nV = len(self.nodes)
        self.attrs = sorted({t.rsplit("-", 1)[0] for t in self.nodes})
        self.r = len(self.attrs)
        rules_ant = [[self.node_idx[t] for t in R["antecedent_tokens"]] for R in fkg.rules]
        rules_out = [self.class_tokens.index(R["consequent_token"]) for R in fkg.rules]
        self.rule_ids = [R.get("id", k) for k, R in enumerate(fkg.rules)]
        self.rule_pos = {rid: k for k, rid in enumerate(self.rule_ids)}
        nR = len(rules_ant); self.nR = nR
        self.rule_lab = np.array(rules_out)
        self.rule_group = np.array([R.get("group", -1 - k) for k, R in enumerate(fkg.rules)])
        self.R_l = [np.where(self.rule_lab == l)[0] for l in range(self.nL)]
        # (3.5) trọng số cạnh a_ii' và (3.6) phân bố nhãn b_il
        Acount = np.zeros((self.nV, self.nV)); cnt = np.zeros(self.nV); bc = np.zeros((self.nV, self.nL))
        for ant, out in zip(rules_ant, rules_out):
            S = sorted(set(ant))
            for i in S:
                cnt[i] += 1; bc[i, out] += 1
                for j in S:
                    if i != j:
                        Acount[i, j] += 1
        self.a = Acount / nR
        self.a_i = self.a.sum(axis=1); self.a_tot = self.a.sum()
        if self.a_tot <= 0:
            raise ValueError("FKG-E: không có cạnh nào (luật cần ít nhất hai điều kiện tiền đề)")
        self.Pn = self.a_i / self.a_tot
        I, J = np.nonzero(self.a)
        self.E_I, self.E_J, self.E_a = I, J, self.a[I, J]
        self.V_R = np.where(cnt > 0)[0]
        b = bc[self.V_R] / cnt[self.V_R][:, None]
        self.b_dist = b
        self.btil = (1 - self.eps_b) * b + self.eps_b / self.nL
        # (3.pool) ma trận gộp phần tiền đề |R| x |𝒱| và hệ số của véc-tơ nhãn
        Pm = np.zeros((nR, self.nV)); coef = np.zeros(nR)
        for k, ant in enumerate(rules_ant):
            a_ = self.alpha if self.alpha is not None else len(ant) / (len(ant) + 1)
            for i in ant:
                Pm[k, i] += a_ / len(ant)
            coef[k] = 1 - a_
        self.Pm, self.out_coef = Pm, coef

    def _query_matrix(self, samples):
        """(3.hj)-(3.query): hàng i là hệ số μ_l(x_j)/r của token (j, l)."""
        Q = np.zeros((len(samples), self.nV))
        for n, s in enumerate(samples):
            for t, deg in s["membership"].items():
                if deg > 0 and t in self.node_idx:
                    Q[n, self.node_idx[t]] += deg / self.r
        return Q

    # ======================================================================== các thành phần mất mát
    def _mask(self, samples):
        """G[i, k] = True nếu luật k thuộc cùng nhóm với mẫu i (luật do chính mẫu sinh ra hoặc bản sao của nó).
        Khi cơ sở luật được sinh từ chính các mẫu huấn luyện, các luật này bị loại khỏi suy diễn cho mẫu đó để
        tránh rò rỉ nhãn (tương tự tính chéo theo phần của Định nghĩa 3.18)."""
        g = np.array([s.get("group", -10**9) for s in samples])
        G = g[:, None] == self.rule_group[None, :]
        return G if G.any() else None

    def _rule_emb(self, params):
        """(3.pool): r_k = (α/r) Σ z + (1 − α) u_out."""
        Z, _, U = params
        return anp.dot(self.Pm, Z) + self.out_coef[:, None] * U[self.rule_lab]

    def _scores(self, params, Q, G=None):
        q = anp.dot(Q, params[0]); Rm = self._rule_emb(params)
        S = _cos(q, Rm, self.eps0)
        if G is not None:
            S = S - 1e6 * G
        sbar = []
        for l in range(self.nL):
            idx = self.R_l[l]
            if len(idx) == 0:
                sbar.append(-anp.ones(Q.shape[0]))
            else:
                sbar.append(self.tau_c * logsumexp(S[:, idx] / self.tau_c, axis=1))
        sbar = anp.stack(sbar, axis=1)
        logits = sbar / self.tau
        pE = (1 - self.eps) * anp.exp(logits - logsumexp(logits, axis=1, keepdims=True)) + self.eps / self.nL
        return S, pE

    def _structural(self, params, eI=None, aI=None):
        """λ_E L_edge + λ_N L_node. eI: chỉ số lô cạnh trong E⁺ (None = toàn bộ); ước lượng không chệch với hệ số
        |E⁺|/b_p. Tham số aI giữ để tương thích, không dùng."""
        Z, Cc, U = params
        lam = self.lam; tot = 0.0
        nE = len(self.E_I)
        if lam["E"] > 0 and nE:
            if eI is None:
                eI = np.arange(nE); se = 1.0
            else:
                se = nE / len(eI)
            I, J, a = self.E_I[eI], self.E_J[eI], self.E_a[eI]
            x = anp.sum(Z[I] * Cc[J], axis=1)
            neg = anp.dot(_log_sigmoid(-anp.dot(Z[I], Cc.T)), self.Pn)       # E_{n~P_n} log σ(−z_iᵀc_n)
            tot = tot + lam["E"] * se * (-anp.sum(a * (_log_sigmoid(x) + self.K * neg)) / self.a_tot)
        if lam["N"] > 0:
            lg = anp.dot(Z[self.V_R], U.T)
            logb = lg - logsumexp(lg, axis=1, keepdims=True)
            tot = tot + lam["N"] * (-anp.sum(self.btil * logb) / len(self.V_R))
        return tot

    def _instance(self, params, Q, y, pF, eF, yF, G=None):
        """λ_R L_rule + λ_I L_inf + λ_P L_pred, trung bình trên các mẫu của Q."""
        S, pE = self._scores(params, Q, G)
        lam = self.lam; tot = 0.0; n = Q.shape[0]
        logpE = anp.log(pE)
        if lam["P"] > 0:
            tot = tot + lam["P"] * (-anp.sum(logpE[np.arange(n), y]) / n)
        if lam["I"] > 0 and pF is not None:
            tot = tot + lam["I"] * anp.sum(pF * (np.log(pF + 1e-300) - logpE)) / n
        if lam["R"] > 0 and eF is not None:
            acc = 0.0
            for l in range(self.nL):
                rows = np.where(yF == l)[0]; idx = self.R_l[l]
                if len(rows) == 0 or len(idx) == 0:
                    continue
                Sl = S[rows][:, idx] / self.tau_c
                logw = Sl - logsumexp(Sl, axis=1, keepdims=True)
                e = eF[rows][:, idx]
                acc = acc + anp.sum(e * (np.log(np.where(e > 0, e, 1.0)) - logw))
            tot = tot + lam["R"] * acc / n
        return tot

    def _reg(self, params):
        return self.lam["C"] * sum(anp.sum(p * p) for p in params)

    def loss(self, params=None, Q=None, y=None, pF=None, eF=None, yF=None, eI=None, aI=None, G=None):
        """Hàm mục tiêu (3.28) — toàn bộ (mặc định) hoặc ước lượng trên lô."""
        params = self.params if params is None else params
        tot = self._structural(params, eI, aI) + self._reg(params)
        if Q is not None and Q.shape[0] > 0:
            tot = tot + self._instance(params, Q, y, pF, eF, yF, G)
        return tot

    def components(self, params=None, data=None):
        """Giá trị từng thành phần (không nhân trọng số) — để kiểm thử và báo cáo."""
        params = self.params if params is None else params
        saved = dict(self.lam); out = {}
        try:
            for name, keys in [("edge", {"E": 1}), ("node", {"N": 1})]:
                self.lam = {k: 0.0 for k in saved}; self.lam.update(keys)
                out[name] = float(self._structural(params))
            if data is not None:
                Q, y, pF, eF, yF = data[:5]
                G = data[5] if len(data) > 5 else None
                for name, keys in [("pred", {"P": 1}), ("inf", {"I": 1}), ("rule", {"R": 1})]:
                    self.lam = {k: 0.0 for k in saved}; self.lam.update(keys)
                    out[name] = float(self._instance(params, Q, y, pF, eF, yF, G))
        finally:
            self.lam = saved
        return out

    # ======================================================================== giáo viên (Định nghĩa 3.18)
    def _teacher(self, fisa, samples, val):
        n = len(samples); yF = np.zeros(n, dtype=int); eF = np.zeros((n, self.nR)); Dlog = np.zeros((n, self.nL))
        for i, s in enumerate(samples):
            D, _, matched = fisa.decision_values(s["membership"])
            yF[i] = self.class_tokens.index(fisa._decide(D, matched))
            for rid, a in fisa.attribution(s["membership"]).items():
                eF[i, self.rule_pos[rid]] = a
            Dlog[i] = [math.log(D[c] + 1e-12) for c in self.class_tokens]
        beta = np.zeros(self.nL)
        if self.nL == 2 and fisa.threshold is not None and fisa.decision == "calibrated":
            beta[1] = -fisa.threshold
        # chọn nhiệt độ T cực tiểu log-mất mát trên tập xác thực
        Dv = np.array([[math.log(fisa.decision_values(s["membership"])[0][c] + 1e-12) for c in self.class_tokens]
                       for s in val]) if val else None
        yv = np.array([self.class_tokens.index(s["label"]) for s in val]) if val else None
        best_T, best = 1.0, np.inf
        for T in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            if Dv is None:
                break
            z = (Dv + beta) / T; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            nll = -np.mean(np.log(p[np.arange(len(yv)), yv] + 1e-12))
            if nll < best:
                best, best_T = nll, T
        z = (Dlog + beta) / best_T; pF = np.exp(z - z.max(1, keepdims=True)); pF /= pF.sum(1, keepdims=True)
        self.teacher_T, self.teacher_beta = best_T, beta
        return pF, eF, yF

    # ======================================================================== Thuật toán 3.10
    def _teacher_from_arrays(self, td, val_labels):
        """Giáo viên tính sẵn bên ngoài (ví dụ tính chéo theo phần): td = {Dlog, eF, yF, Dlog_val, beta}."""
        beta = np.asarray(td.get("beta", np.zeros(self.nL)), dtype=float)
        best_T, best = 1.0, np.inf
        Dv = td.get("Dlog_val")
        if Dv is not None and len(Dv):
            yv = np.array([self.class_tokens.index(l) for l in val_labels])
            for T in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
                z = (Dv + beta) / T; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                nll = -np.mean(np.log(p[np.arange(len(yv)), yv] + 1e-12))
                if nll < best:
                    best, best_T = nll, T
        z = (np.asarray(td["Dlog"]) + beta) / best_T
        pF = np.exp(z - z.max(1, keepdims=True)); pF /= pF.sum(1, keepdims=True)
        self.teacher_T, self.teacher_beta = best_T, beta
        return pF, np.asarray(td["eF"]), np.asarray(td["yF"], dtype=int)

    def fit(self, fisa_model=None, train_samples=None, val_samples=None, batch_size=None, teacher_data=None):
        t0 = time.time()
        rng = np.random.RandomState(self.seed)
        if batch_size:
            self.b = int(batch_size)
        train = list(train_samples or [])
        if val_samples is None and train:
            perm = rng.permutation(len(train)); nv = max(1, int(round(self.val_fraction * len(train))))
            val = [train[i] for i in perm[:nv]]; train = [train[i] for i in perm[nv:]]
        else:
            val = list(val_samples or [])
        self.teacher = fisa_model
        if fisa_model is None and teacher_data is None and (self.lam["I"] > 0 or self.lam["R"] > 0):
            warnings.warn("FKG-E: không có FISA giáo viên — bỏ L_inf và L_rule (λ_I = λ_R = 0).")
            self.lam["I"] = self.lam["R"] = 0.0
        Q = self._query_matrix(train); y = np.array([self.class_tokens.index(s["label"]) for s in train], dtype=int)
        pF = eF = yF = None
        if teacher_data is not None:
            pF, eF, yF = self._teacher_from_arrays(teacher_data, [s["label"] for s in val])
        elif fisa_model is not None and train:
            pF, eF, yF = self._teacher(fisa_model, train, val)
        G = self._mask(train) if train else None
        Gv = self._mask(val) if val else None
        Qv = self._query_matrix(val) if val else None
        yv = np.array([self.class_tokens.index(s["label"]) for s in val], dtype=int) if val else None
        self._train_data = (Q, y, pF, eF, yF)
        self._train_mask = G

        full = lambda p: self.loss(p, Q, y, pF, eF, yF, G=G)
        L0 = float(full(self.params))
        self.B0 = math.sqrt(L0 / self.lam["C"])                    # Mệnh đề 3.coer
        g_batch = grad(lambda p, Qb, yb, pFb, eFb, yFb, eI, aI, Gb: self.loss(p, Qb, yb, pFb, eFb, yFb, eI, aI, Gb))
        g_full = grad(full)
        N = len(train); B_ep = max(1, int(math.ceil(N / self.b))) if N else 1
        m = [np.zeros_like(p) for p in self.params]; v = [np.zeros_like(p) for p in self.params]
        vhat = [np.zeros_like(p) for p in self.params]; b1, b2 = 0.9, 0.999; step = 0
        best_val, best_params, wait = np.inf, [p.copy() for p in self.params], 0
        nE, nA = len(self.E_I), 0
        self.stop_reason = "T_ep"
        for ep in range(self.T_ep):
            if self.optimizer == "gd_full":
                # Thuật toán 3.10: giảm gradient toàn bộ có chiếu, bước chọn theo quy tắc giảm đủ (Armijo)
                # L(Θ⁺) ≤ L(Θ) − (η/2)‖∇L(Θ)‖² — đúng bất đẳng thức (i) của Định lý hội tụ.
                gr = g_full(self.params); L_cur = float(full(self.params))
                eta = self.eta_gd if ep == 0 else min(2 * self._last_eta, self.eta_gd)
                ok = False
                for _ in range(40):
                    new = [self.params[k] - eta * gr[k] for k in range(3)]
                    nrm = math.sqrt(sum(float(np.sum(p * p)) for p in new))
                    proj = nrm > self.B0
                    if proj:
                        new = [p * (self.B0 / nrm) for p in new]
                    gm2 = sum(float(np.sum((self.params[k] - new[k]) ** 2)) for k in range(3)) / eta ** 2   # ‖G_η‖²
                    if float(full(new)) <= L_cur - 0.5 * eta * gm2:
                        ok = True; break
                    eta *= 0.5
                self._last_eta = eta; self.history["eta"].append(eta)
                self.history["proj"].append(bool(proj)); self.history["ls_fail"].append(not ok)
                self.params = new
                B_loop = 0
            else:
                B_loop = B_ep
            perm = rng.permutation(N) if (N and B_loop) else np.array([], dtype=int)
            for t in range(B_loop):
                bi = perm[t * self.b:(t + 1) * self.b]
                eI = rng.choice(nE, size=min(self.b_p, nE), replace=False) if nE else None
                aI = None
                sub = lambda a: None if a is None else a[bi]
                gr = g_batch(self.params, Q[bi], y[bi], sub(pF), sub(eF), sub(yF), eI, aI, sub(G))
                step += 1
                if self.optimizer == "amsgrad":
                    new = []
                    for k in range(3):
                        m[k] = b1 * m[k] + (1 - b1) * gr[k]; v[k] = b2 * v[k] + (1 - b2) * gr[k] ** 2
                        vhat[k] = np.maximum(vhat[k], v[k])
                        new.append(self.params[k] - self.eta * m[k] / (np.sqrt(vhat[k]) + 1e-8))
                else:
                    new = [self.params[k] - self.eta * gr[k] for k in range(3)]
                nrm = math.sqrt(sum(float(np.sum(p * p)) for p in new))   # chiếu lên hình cầu bán kính B0
                if nrm > self.B0:
                    new = [p * (self.B0 / nrm) for p in new]
                self.params = new
            gf = g_full(self.params)
            gnorm = math.sqrt(sum(float(np.sum(gg * gg)) for gg in gf))
            self.history["grad_norm"].append(gnorm); self.history["loss"].append(float(full(self.params)))
            if Qv is not None and len(val):
                _, pEv = self._scores(self.params, Qv, Gv)
                vl = float(-np.mean(np.log(pEv[np.arange(len(yv)), yv])))
                self.history["val_pred"].append(vl)
                if vl < best_val - 1e-9:
                    best_val, best_params, wait = vl, [p.copy() for p in self.params], 0
                else:
                    wait += 1
            if gnorm <= self.eps_g:                                   # Hệ quả 3.stop
                self.stop_reason = "grad"; break
            if Qv is not None and wait >= self.patience:
                self.stop_reason = "early"; break
        if Qv is not None and len(val) and best_params is not None:
            self.params = best_params
        self.epochs_run = ep + 1
        self._finalize()
        if self.nL == 2 and Qv is not None and len(val):
            self._calibrate(Qv, yv, Gv)
        self.train_time_s = time.time() - t0
        return self

    def _finalize(self):
        self.rule_emb = np.asarray(self._rule_emb(self.params))

    def _logratio(self, pE):
        return np.log(pE[:, 1]) - np.log(pE[:, 0])

    def _calibrate(self, Qv, yv, Gv=None):
        """τ_E: ngưỡng trên log p_E(1) − log p_E(0) cực đại độ chính xác cân bằng trên tập xác thực."""
        _, pE = self._infer(Qv, Gv); s = self._logratio(pE)
        if len(np.unique(yv)) < 2:
            self.threshold = 0.0; return
        cand = np.unique(np.concatenate([[0.0], np.quantile(s, np.linspace(0.01, 0.99, 197))]))
        best, bt = -1, 0.0
        for t in cand:
            p = (s > t).astype(int); v = 0.5 * (np.mean(p[yv == 1] == 1) + np.mean(p[yv == 0] == 0))
            if v > best + 1e-12:
                best, bt = v, float(t)
        self.threshold = bt

    # ======================================================================== Thuật toán 3.11
    def _infer(self, Q, G=None):
        """Thuật toán 3.11: q -> s_k (một phép nhân |R| x d_e với r_k tính sẵn) -> s̄_l -> p_E."""
        q = Q @ self.params[0]
        S = np.asarray(_cos(q, self.rule_emb, self.eps0))
        if G is not None:
            S = S - 1e6 * G
        sbar = np.full((Q.shape[0], self.nL), -1.0)
        for l in range(self.nL):
            idx = self.R_l[l]
            if len(idx):
                z = S[:, idx] / self.tau_c; mx = z.max(axis=1, keepdims=True)
                sbar[:, l] = self.tau_c * (mx[:, 0] + np.log(np.exp(z - mx).sum(axis=1)))
        z = sbar / self.tau; z = z - z.max(axis=1, keepdims=True)
        pE = (1 - self.eps) * np.exp(z) / np.exp(z).sum(axis=1, keepdims=True) + self.eps / self.nL
        return S, pE

    def _decide(self, pE):
        if self.nL == 2 and self.threshold is not None:
            return np.where(self._logratio(pE) > self.threshold, 1, 0)
        return np.argmax(pE, axis=1)

    def predict_one(self, membership):
        t0 = time.time()
        Q = self._query_matrix([{"membership": membership}])
        _, pE = self._infer(Q)
        yh = self.class_tokens[int(self._decide(pE)[0])]
        return {c: float(pE[0, i]) for i, c in enumerate(self.class_tokens)}, yh, time.time() - t0

    def explain_one(self, membership, K_act=5):
        """a_E (Định nghĩa 3.attrE) và K_act luật có đóng góp lớn nhất."""
        Q = self._query_matrix([{"membership": membership}])
        S, pE = self._infer(Q)
        l = int(self._decide(pE)[0]); idx = self.R_l[l]
        z = S[0, idx] / self.tau_c; w = np.exp(z - z.max()); w /= w.sum()
        aE = {self.rule_ids[k]: float(wk) for k, wk in zip(idx, w)}
        top = sorted(aE.items(), key=lambda kv: -kv[1])[:K_act]
        return aE, top

    def evaluate(self, samples, fisa_model=None):
        fisa = fisa_model or self.teacher
        t0 = time.time()
        Q = self._query_matrix(samples)
        y = np.array([self.class_tokens.index(s["label"]) for s in samples])
        times = []
        for n in range(len(samples)):                                # đo thời gian theo từng truy vấn
            t1 = time.time(); self._infer(Q[n:n + 1]); times.append(time.time() - t1)
        S, pE = self._infer(Q); yh = self._decide(pE)
        from models.fisa import _macro_f1, _balanced_acc, _auc
        yt = [self.class_tokens[i] for i in y]; yp = [self.class_tokens[i] for i in yh]
        out = {"accuracy": float(np.mean(yh == y)), "f1_macro": _macro_f1(yt, yp, self.class_tokens),
               "balanced_accuracy": _balanced_acc(yt, yp, self.class_tokens),
               "log_loss": float(-np.mean(np.log(pE[np.arange(len(y)), y]))),
               "avg_time_per_query_ms": float(np.mean(times) * 1000), "total_time_s": time.time() - t0,
               "threshold": self.threshold, "y_true": yt, "y_pred": yp}
        if self.nL == 2:
            out["auc"] = _auc(y.tolist(), self._logratio(pE).tolist())
        if fisa is not None:                                         # (R2), (R3)
            agree, dev = [], []
            for n, s in enumerate(samples):
                D, _, matched = fisa.decision_values(s["membership"])
                yF = fisa._decide(D, matched); agree.append(yF == yp[n])
                aF = fisa.attribution(s["membership"])
                l = int(yh[n]); idx = self.R_l[l]; z = S[n, idx] / self.tau_c
                w = np.exp(z - z.max()); w /= w.sum()
                aE = {self.rule_ids[k]: wk for k, wk in zip(idx, w)}
                keys = set(aE) | set(aF)
                dev.append(0.5 * sum(abs(aE.get(k, 0.0) - aF.get(k, 0.0)) for k in keys))
            out["agreement_fisa"] = float(np.mean(agree)); out["dev_rule"] = float(np.mean(dev))
        return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples
    from models.fisa import FISA
    fkg = generate_synthetic_fkg(n_rules=200, seed=1)
    train = generate_synthetic_test_samples(fkg, n_samples=200, seed=10)
    test = generate_synthetic_test_samples(fkg, n_samples=150, seed=11)
    fisa = FISA(fkg).fit(val_samples=train)
    print("FISA :", {k: round(v, 4) for k, v in fisa.evaluate(test).items() if k in ("accuracy", "balanced_accuracy", "auc")})
    m = FKGE(fkg, T_ep=30).fit(fisa_model=fisa, train_samples=train)
    r = m.evaluate(test)
    print("FKG-E:", {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
          "dừng:", m.stop_reason, "epoch:", m.epochs_run)
