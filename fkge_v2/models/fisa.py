"""
models/fisa.py — Suy luận FISA (Fuzzy Inference by Similarity Aggregation),
đúng công thức (1.18)-(1.20) gốc (`\\cite{ref7}`), đã hiệu chỉnh và nhắc lại
trong Chương 3 (Mục "Nhắc lại ký hiệu suy luận FISA"):

    W_{i,v,l} = sum_{t: Val_t(P_i)=v} B^t_il          (3.67, bảng 3 chiều)
    C_il(x)   = W_{i, x_i, l}                          (3.68, TRA CỨU trực tiếp)
    D_l(x)    = min_i C_il(x) + max_i C_il(x)          (3.69 = 1.19 gốc)
    y_hat     = argmax_l D_l(x)                        (3.70 = 1.20 gốc)

SỬA QUAN TRỌNG so với phiên bản trước: C_il(x) KHÔNG còn là tổng có trọng
số mờ liên tục qua mọi giá trị ngôn ngữ v (tức KHÔNG dùng mu_v(x_i) nhân
vào), vì số hạng đó KHÔNG xuất hiện trong công thức gốc (1.18). Thay vào
đó, với mỗi thuộc tính i, chỉ giá trị ngôn ngữ THẮNG CUỘC (độ thuộc mờ lớn
nhất) mới được dùng để TRA CỨU vào bảng W — đúng cách FISA gốc thực hiện
so khớp rời rạc trên giá trị đã mờ hoá, không phải cộng dồn liên tục.

Vì luật đã cho ở dạng (antecedent_tokens -> consequent_token, support,
confidence) thay vì ma trận A/B tường minh theo Công thức (1.17)/(3.9),
FISA ở đây suy ra W trực tiếp từ tập luật: mỗi luật "bỏ phiếu" cho token
tiền đề của nó (đóng vai trò Val_t(P_i)) với trọng số = support*confidence
của luật t, cộng dồn vào W[i][v][l]. Đây là một cách hiện thực hoá xấp xỉ
của B^t_il (dùng support*confidence thay cho công thức đầy đủ qua A), có
thể thay bằng B thật (Công thức 3.9) nếu bạn đã có sẵn ma trận A/B từ
pipeline khai phá luật của Chương 2/3.
"""
import time
import math
from collections import defaultdict, Counter

from data.fkg_io import token_attribute


class FISA:
    def __init__(self, fkg, inference_mode="lookup"):
        if inference_mode not in {"lookup", "sequential"}:
            raise ValueError("inference_mode must be 'lookup' or 'sequential'.")
        self.fkg = fkg
        self.inference_mode = inference_mode
        self.class_tokens = fkg.class_tokens
        self.W = None          # W[attr][v][l] -> float  (bảng ba chiều)
        self._attr_of = {}      # token -> tên thuộc tính (phần trước dấu '-')
        self._values_of_attr = defaultdict(set)  # attr -> {v1, v2, ...} (= V_i)
        for t in fkg.vocab:
            if not t.startswith("class-"):
                attr = token_attribute(t)
                self._attr_of[t] = attr
                self._values_of_attr[attr].add(t)

    def fit(self):
        """Xây bảng ba chiều W_{i,v,l} theo (3.67): với mỗi luật t và mỗi
        token tiền đề v (đóng vai trò Val_t(P_i)=v tại thuộc tính i mà v
        thuộc về), cộng trọng số luật vào đúng ô (i, v, l). Việc CHỈ cộng
        vào token v cụ thể (không phải mọi token của thuộc tính i) chính là
        cơ chế lọc "chỉ luật có Val_t(P_i)=v mới đóng góp" -- khớp đúng
        (3.67), không cần thêm điều kiện lọc tường minh nào khác."""
        t0 = time.time()
        W = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # W[attr][v][l]
        for r in self.fkg.rules:
            w = r["support"] * r["confidence"]
            l = r["consequent_token"]
            for v in r["antecedent_tokens"]:
                attr = self._attr_of.get(v)
                if attr is not None:
                    W[attr][v][l] += w
        self.W = W
        self.fit_time = time.time() - t0
        return self

    def _dominant_value(self, membership, attr):
        """Tìm giá trị ngôn ngữ THẮNG CUỘC (độ thuộc lớn nhất) của thuộc
        tính `attr` trong `membership` -- đây là x_i sau khi mờ hoá và chọn
        nhãn thắng, dùng để TRA CỨU (3.68), không phải để nhân trọng số."""
        best_v, best_deg = None, -1.0
        for v in self._values_of_attr.get(attr, ()):
            deg = membership.get(v, 0.0)
            if deg > best_deg:
                best_v, best_deg = v, deg
        if best_deg <= 0:
            return None  # thuộc tính này không có bằng chứng nào trong bản ghi
        return best_v

    def _C(self, membership):
        """C_il(x) theo (3.68) ĐÃ SỬA: với mỗi thuộc tính i có bằng chứng
        trong membership, tra cứu DUY NHẤT giá trị thắng cuộc x_i vào bảng
        W, KHÔNG cộng dồn/nhân trọng số qua các giá trị khác của thuộc tính
        đó."""
        C = defaultdict(lambda: defaultdict(float))  # C[attr][class]
        for attr in self.W:
            v_star = self._dominant_value(membership, attr)
            if v_star is None:
                continue
            for l, w in self.W[attr][v_star].items():
                C[attr][l] = w   # TRA CỨU trực tiếp -- không cộng dồn qua v
        return C

    def _C_sequential(self, membership):
        dominant = {
            attr: self._dominant_value(membership, attr)
            for attr in self._values_of_attr
        }
        C = defaultdict(lambda: defaultdict(float))
        for rule in self.fkg.rules:
            class_token = rule["consequent_token"]
            weight = rule["support"] * rule["confidence"]
            for token in rule["antecedent_tokens"]:
                attr = self._attr_of.get(token)
                if attr is not None and dominant.get(attr) == token:
                    C[attr][class_token] += weight
        return C

    def predict_one(self, membership):
        """Trả về (D_l dict, nhãn dự đoán, thời gian suy diễn)."""
        t0 = time.time()
        C = (self._C(membership) if self.inference_mode == "lookup"
             else self._C_sequential(membership))
        D = defaultdict(float)
        if len(C) == 0:
            elapsed = time.time() - t0
            return {}, None, elapsed
        for l in self.class_tokens:
            vals = [C[attr].get(l, 0.0) for attr in C]
            if not vals:
                D[l] = 0.0
            else:
                D[l] = min(vals) + max(vals)
        y_hat = max(D, key=D.get)
        elapsed = time.time() - t0
        return dict(D), y_hat, elapsed

    def predict_proba_one(self, membership, temperature=1.0):
        """Chuyển D_l thành phân phối xác suất bằng softmax (dùng làm
        p^F_cal trong Mục 3.4.5 để tính KL-divergence với FKG-E)."""
        D, y_hat, elapsed = self.predict_one(membership)
        if not D:
            n = len(self.class_tokens)
            return {c: 1.0 / n for c in self.class_tokens}, y_hat, elapsed
        vals = [D.get(c, 0.0) for c in self.class_tokens]
        m = max(vals)
        exps = [math.exp((v - m) / max(temperature, 1e-6)) for v in vals]
        s = sum(exps) + 1e-12
        proba = {c: e / s for c, e in zip(self.class_tokens, exps)}
        return proba, y_hat, elapsed

    def evaluate(self, samples):
        """Chạy suy diễn trên toàn bộ tập mẫu và trả về dự đoán có điểm số."""
        from models.metrics import classification_metrics

        y_true, y_pred, probabilities = [], [], []
        total_time = 0.0
        per_query_times = []
        for s in samples:
            proba, y_hat, elapsed = self.predict_proba_one(s["membership"])
            total_time += elapsed
            per_query_times.append(elapsed)
            y_true.append(s["label"])
            y_pred.append(y_hat if y_hat is not None else "NONE")
            probabilities.append(proba)
        positive_class = self.class_tokens[-1]
        y_score = [proba.get(positive_class, 0.0) for proba in probabilities]
        result = classification_metrics(y_true, y_pred, y_score, self.class_tokens)
        result.update({
            "total_time_s": total_time,
            "avg_time_per_query_ms": (total_time / len(samples)) * 1000,
            "fit_time_s": getattr(self, "fit_time", 0.0),
            "y_true": y_true, "y_pred": y_pred,
            "y_score": y_score, "probabilities": probabilities,
            "inference_mode": self.inference_mode,
        })
        return result


def _macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def majority_class_baseline(samples):
    """Accuracy của bộ dự đoán ngây thơ 'luôn đoán lớp xuất hiện nhiều nhất
    trong tập test'. MỌI mô hình phải được so sánh với con số này -- nếu một
    mô hình không vượt qua rõ rệt baseline này, đó là dấu hiệu mô hình đang
    "sụp" về việc đoán theo lớp đa số (majority-class collapse), một lỗi rất
    dễ xảy ra khi tín hiệu học biểu diễn (không giám sát) áp đảo tín hiệu
    phân loại (có giám sát) trong hàm mất mát tổng hợp."""
    labels = [s["label"] for s in samples]
    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]
    return top_count / len(samples), top_label


def warn_if_not_beating_majority(result, samples, model_name="Mô hình", margin=0.02):
    """Flag class collapse using balanced accuracy, not raw accuracy alone."""
    base_acc, base_label = majority_class_baseline(samples)
    acc = result["accuracy"]
    bal_acc = result.get("balanced_accuracy", 0.5)
    if bal_acc <= 0.52:
        print(f"  !! CẢNH BÁO: {model_name} có BalAcc={bal_acc:.4f}; "
              f"cần kiểm tra dự đoán một lớp (accuracy={acc:.4f}, "
              f"baseline đa số={base_acc:.4f}, lớp '{base_label}').")
        return False
    if acc <= base_acc + margin:
        print(f"  {model_name}: accuracy={acc:.4f} gần hoặc dưới mốc lớp đa số "
              f"{base_acc:.4f}, nhưng BalAcc={bal_acc:.4f}; "
              "đọc AUC/BalAcc/F1 thay cho accuracy đơn lẻ.")
    return True


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples

    fkg = generate_synthetic_fkg(n_rules=200, seed=1)
    samples = generate_synthetic_test_samples(fkg, n_samples=200, seed=2)
    fisa = FISA(fkg).fit()
    res = fisa.evaluate(samples)
    print(f"FISA (đã sửa theo 1.18-1.20) trên dữ liệu tổng hợp: "
          f"Accuracy={res['accuracy']:.4f}, F1={res['f1_macro']:.4f}, "
          f"TG suy diễn TB={res['avg_time_per_query_ms']:.4f} ms")
    base_acc, base_label = majority_class_baseline(samples)
    print(f"Baseline lớp đa số: {base_acc:.4f}")
    assert res["accuracy"] > base_acc - 0.05, (
        "FISA (đã sửa) cho accuracy thấp hơn đáng kể baseline lớp đa số -- "
        "kiểm tra lại logic tra cứu dominant-value!"
    )
    print("Kiểm thử FISA (bản đã sửa theo 1.18-1.20): OK")
