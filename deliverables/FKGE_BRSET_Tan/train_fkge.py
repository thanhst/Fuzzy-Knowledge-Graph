"""Float64 exact full-gradient FKG-E with weighted duplicate rules and blocks."""
from __future__ import annotations

import gc
import math
import time

import numpy as np

from compact_fkge import CompactFISA, threshold_on_validation


def build_teacher(prepared, directory):
    R, y, weights = prepared["R"], prepared["y"], prepared["weights"]
    started = time.time()
    full = CompactFISA(R, y, weights)
    Dv, mv = full.log_values(prepared["Rval"])
    score = np.where(mv, Dv[:, 1] - Dv[:, 0], 0.0)
    tau = threshold_on_validation(prepared["yval"], score)
    beta = np.array([0.0, -tau])
    best_T, best_loss = 1.0, np.inf
    for temperature in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        z = (Dv + beta) / temperature
        probabilities = np.exp(z - z.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        loss = -np.mean(np.log(probabilities[np.arange(len(z)), prepared["yval"]] + 1e-12))
        if loss < best_loss:
            best_loss, best_T = loss, temperature
    n = len(R)
    targets = np.lib.format.open_memmap(directory / "teacher_attribution.npy", mode="w+",
                                         dtype=np.float64, shape=(n, n))
    targets[:] = 0
    log_values = np.empty((n, 2))
    chosen = np.empty(n, dtype=int)
    for part in range(5):
        query = np.flatnonzero(prepared["parts"] == part)
        rules = np.flatnonzero(prepared["parts"] != part)
        if not len(query):
            continue
        if not len(rules):
            raise ValueError("Cross-fitting unit occupies the complete training graph")
        teacher = CompactFISA(R[rules], y[rules], weights[rules])
        D, matched = teacher.log_values(R[query])
        prediction = np.where(matched, (D[:, 1] - D[:, 0] > tau).astype(int), teacher.fallback)
        log_values[query] = D
        chosen[query] = prediction
        for start in range(0, len(query), 64):
            indices = query[start:start + 64]
            a = teacher.attribution(R[indices], prediction[start:start + 64])
            targets[np.ix_(indices, rules)] = a
        targets.flush()
        del teacher
        gc.collect()
        print(f"TEACHER part={part + 1}/5 unique_queries={len(query)} elapsed={time.time() - started:.1f}s", flush=True)
    z = (log_values + beta) / best_T
    pF = np.exp(z - z.max(1, keepdims=True))
    pF /= pF.sum(1, keepdims=True)
    np.savez(directory / "teacher_small.npz", pF=pF, yF=chosen, Dlog=log_values,
             threshold=tau, temperature=best_T, validation_log_values=Dv,
             validation_matched=mv, seconds=time.time() - started)
    del targets
    return full, time.time() - started


class CompactFKGE:
    def __init__(self, R, y, weights, units, pF, eF, yF, *, alpha=0.7, d_e=32,
                 lam_I=1.0, lam_P=1.0, seed=42, device="cuda", block_size=256):
        global torch
        import torch
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.device, self.dtype = device, torch.float64
        self.tensor = lambda x, dtype=None: torch.as_tensor(x, dtype=dtype or self.dtype, device=device)
        self.R, self.y_np, self.weights_np = R, y, weights
        self.units_np, self.eF = units, eF
        self.alpha, self.d_e, self.lam_I, self.lam_P = alpha, d_e, lam_I, lam_P
        self.block_size = block_size
        self.r = R.shape[1]
        self.node_codes = np.unique((np.arange(self.r)[None, :] * 5 + R).ravel())
        self.nV = len(self.node_codes)
        self.tau, self.tau_c, self.eps, self.eps0 = 0.1, 0.1, 0.05, 1e-6
        self.lam_C = 1e-3
        H = self._one_hot(R)
        counts = H.T @ weights
        A = (H.T * weights) @ H / weights.sum()
        np.fill_diagonal(A, 0.0)
        I, J = np.nonzero(A)
        self.I, self.J = self.tensor(I, torch.long), self.tensor(J, torch.long)
        self.edge_a, self.a_total = self.tensor(A[I, J]), float(A.sum())
        self.Pn = self.tensor(A.sum(1) / A.sum())
        active = np.flatnonzero(counts > 0)
        bc = H.T @ (weights[:, None] * np.eye(2)[y])
        self.active = self.tensor(active, torch.long)
        self.btil = self.tensor(0.95 * bc[active] / counts[active, None] + 0.05 / 2)
        self.H = self.tensor(H / self.r)
        self.log_counts = self.tensor(np.log(weights))
        self.units = self.tensor(units, torch.long)
        self.y = self.tensor(y, torch.long)
        self.yF = self.tensor(yF, torch.long)
        self.pF = self.tensor(pF)
        self.query_weights = self.tensor(weights / weights.sum())
        self.class_indices = [self.tensor(np.flatnonzero(y == label), torch.long) for label in (0, 1)]
        rng = np.random.RandomState(seed)
        self.params = [self.tensor(rng.normal(0, 0.1, shape)).requires_grad_(True)
                       for shape in [(self.nV, d_e), (self.nV, d_e), (2, d_e)]]
        self.history = {key: [] for key in ["loss", "val_pred", "grad_norm", "eta", "proj", "ls_fail"]}
        self.threshold = 0.0

    def _one_hot(self, R):
        H = np.zeros((len(R), R.shape[1] * 5))
        H[np.arange(len(R))[:, None], np.arange(R.shape[1])[None, :] * 5 + R] = 1.0
        return H[:, self.node_codes]

    def _structural(self, params):
        Z, C, U = params
        positive = (Z[self.I] * C[self.J]).sum(1)
        negative = torch.nn.functional.logsigmoid(-(Z[self.I] @ C.T)) @ self.Pn
        edge = -(self.edge_a * (torch.nn.functional.logsigmoid(positive) + 5 * negative)).sum() / self.a_total
        node = -(self.btil * torch.log_softmax(Z[self.active] @ U.T, dim=1)).sum() / len(self.active)
        return edge + node + self.lam_C * sum((p * p).sum() for p in params)

    def _scores(self, params, H, groups=None):
        Z, _, U = params
        query = H @ Z
        rules = self.alpha * (self.H @ Z) + (1 - self.alpha) * U[self.y]
        similarity = (query @ rules.T) / (
            torch.sqrt((query * query).sum(1) + self.eps0)[:, None]
            * torch.sqrt((rules * rules).sum(1) + self.eps0)[None, :])
        if groups is not None:
            similarity = similarity - 1e6 * (groups[:, None] == self.units[None, :])
        logits, log_weights = [], []
        for indices in self.class_indices:
            z = similarity[:, indices] / self.tau_c + self.log_counts[indices]
            normalization = torch.logsumexp(z, dim=1)
            logits.append(self.tau_c * normalization)
            log_weights.append(z - normalization[:, None])
        logits = torch.stack(logits, dim=1) / self.tau
        probabilities = (1 - self.eps) * torch.softmax(logits, dim=1) + self.eps / 2
        return probabilities, log_weights

    def _instance(self, params, start, stop):
        probability, log_weights = self._scores(params, self.H[start:stop], self.units[start:stop])
        log_probability = torch.log(probability)
        local_y = self.y[start:stop]
        loss = -self.lam_P * log_probability[torch.arange(stop - start, device=self.device), local_y]
        pF = self.pF[start:stop]
        loss = loss + self.lam_I * (pF * (torch.log(pF + 1e-300) - log_probability)).sum(1)
        e = self.tensor(np.array(self.eF[start:stop], copy=True))
        local_yF = self.yF[start:stop]
        for label, indices in enumerate(self.class_indices):
            target = e[:, indices]
            kl = (target * (torch.log(torch.where(target > 0, target, 1.0)) - log_weights[label])).sum(1)
            loss = loss + torch.where(local_yF == label, kl, 0.0)
        return (loss * self.query_weights[start:stop]).sum()

    def value_grad(self, params=None):
        params = self.params if params is None else params
        for p in params:
            p.grad = None
        structural = self._structural(params)
        structural.backward()
        value = structural.detach()
        for start in range(0, len(self.R), self.block_size):
            loss = self._instance(params, start, min(start + self.block_size, len(self.R)))
            loss.backward()
            value = value + loss.detach()
        return float(value.item()), [p.grad.detach().clone() for p in params]

    def value(self, params=None):
        params = self.params if params is None else params
        with torch.no_grad():
            value = self._structural(params)
            for start in range(0, len(self.R), self.block_size):
                value = value + self._instance(params, start, min(start + self.block_size, len(self.R)))
            return float(value.item())

    def probabilities(self, R, params=None):
        params = self.params if params is None else params
        result = []
        with torch.no_grad():
            for start in range(0, len(R), self.block_size):
                H = self.tensor(self._one_hot(R[start:start + self.block_size]) / self.r)
                probability, _ = self._scores(params, H)
                result.append(probability.cpu().numpy())
        return np.concatenate(result)

    def fit(self, Rval, yval, max_epochs=200, patience=20, verbose=False):
        if max_epochs < 1 or patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        started = time.time()
        loss, gradient = self.value_grad()
        self.B0 = math.sqrt(loss / self.lam_C)
        best_val, best_params, waiting = np.inf, None, 0
        last_eta = 1.0
        self.stop_reason = "T_ep"
        for epoch in range(max_epochs):
            eta = 1.0 if epoch == 0 else min(2 * last_eta, 1.0)
            success = False
            with torch.no_grad():
                for _ in range(40):
                    candidate = [p.detach() - eta * g for p, g in zip(self.params, gradient)]
                    norm = math.sqrt(sum(float((p * p).sum().item()) for p in candidate))
                    projection = norm > self.B0
                    if projection:
                        candidate = [p * (self.B0 / norm) for p in candidate]
                    mapping_norm2 = sum(float(((p.detach() - q) ** 2).sum().item())
                                        for p, q in zip(self.params, candidate)) / eta ** 2
                    if self.value(candidate) <= loss - 0.5 * eta * mapping_norm2:
                        success = True
                        break
                    eta *= 0.5
            self.params = [p.detach().requires_grad_(True) for p in candidate]
            last_eta = eta
            # Reuse this full gradient at the next iteration: the objective is
            # deterministic and no stochastic minibatch update is performed.
            loss, gradient = self.value_grad()
            norm_gradient = math.sqrt(sum(float((g * g).sum().item()) for g in gradient))
            probability = self.probabilities(Rval)
            val_loss = float(-np.mean(np.log(probability[np.arange(len(yval)), yval])))
            for key, value in [("loss", loss), ("val_pred", val_loss), ("grad_norm", norm_gradient),
                               ("eta", eta), ("proj", projection), ("ls_fail", not success)]:
                self.history[key].append(value)
            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_params = [p.detach().clone() for p in self.params]
                waiting = 0
            else:
                waiting += 1
            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print(f"EPOCH {epoch + 1}: loss={loss:.6f} val={val_loss:.6f} elapsed={time.time() - started:.1f}s", flush=True)
            if norm_gradient <= 1e-4:
                self.stop_reason = "grad"
                break
            if waiting >= patience:
                self.stop_reason = "early"
                break
        if best_params is not None:
            self.params = [p.requires_grad_(True) for p in best_params]
        probability = self.probabilities(Rval)
        score = np.log(probability[:, 1]) - np.log(probability[:, 0])
        self.threshold = threshold_on_validation(yval, score)
        self.val_loss = float(-np.mean(np.log(probability[np.arange(len(yval)), yval])))
        self.epochs = epoch + 1
        self.seconds = time.time() - started
        return self

    def save(self, path):
        np.savez(path, Z=self.params[0].detach().cpu().numpy(),
                 C=self.params[1].detach().cpu().numpy(), U=self.params[2].detach().cpu().numpy(),
                 threshold=self.threshold, alpha=self.alpha, d_e=self.d_e,
                 lam_I=self.lam_I, lam_P=self.lam_P, node_codes=self.node_codes)
