"""
data/fkg_io.py — Định dạng dữ liệu chuẩn cho một cơ sở luật FKG/FKG-MM/FKGS,
và bộ nạp/lưu tương ứng. Cũng cung cấp bộ SINH DỮ LIỆU TỔNG HỢP (synthetic)
để kiểm thử toàn bộ pipeline (KB1-KB6, ablation, baseline) trước khi có dữ
liệu BRSET/Diabetes thật.

ĐỊNH DẠNG FILE LUẬT (JSON), khớp Định nghĩa 3.2 (Chương 3):
{
  "meta": {"dataset": "BRSET", "n_classes": 2, "class_names": ["No-DR","DR"]},
  "vocab": ["Age-Low", "Age-Medium", "Age-High", "HbA1c-Low", ..., "class-DR"],
  "rules": [
      {
        "id": 0,
        "antecedent_tokens": ["Age-High", "HbA1c-High", "LesionArea-High"],
        "consequent_token": "class-DR",
        "confidence": 0.87,
        "support": 0.12
      },
      ...
  ],
  "edges": [   # dùng cho Node Loss (mu_ij, Định nghĩa 3.2.2 / Mục 3.3.4)
      {"u": "Age-High", "v": "HbA1c-High", "mu": 0.63},
      ...
  ]
}

ĐỊNH DẠNG FILE TEST/TRAIN GỘP CHUNG (JSON) — DÙNG CHO K-FOLD:
{
  "samples": [
      {
        "membership": {"Age-High": 0.8, "Age-Medium": 0.2, "HbA1c-High": 0.9, ...},
        "label": "class-DR",
        "patient_id": "P00123"
      },
      ...
  ]
}

QUAN TRỌNG (Mục 3.5.1 Chương 3 — nguyên tắc phân chia dữ liệu và phòng
tránh rò rỉ): trường "patient_id" là BẮT BUỘC. Một bệnh nhân có thể có
nhiều ảnh/bản ghi (nhiều "samples" cùng patient_id) — các bản ghi này
KHÔNG được phép rơi vào cả train và test của cùng một fold. Toàn bộ pipeline
thực nghiệm dùng Stratified Group K-Fold (patient_id làm biến nhóm) thay vì
chia đơn giản theo thứ tự, xem data/kfold_utils.py.
"""
import json
import os
import random
import math


class FKGRuleBase:
    """Cấu trúc dữ liệu trong bộ nhớ cho một cơ sở luật FKG đã khai phá."""

    def __init__(self, vocab, rules, edges, meta=None):
        self.vocab = list(vocab)                       # danh sách token (str)
        self.token2idx = {t: i for i, t in enumerate(self.vocab)}
        self.rules = rules                              # list[dict]
        self.edges = edges                               # list[dict {u,v,mu}]
        self.meta = meta or {}
        self.class_tokens = sorted({r["consequent_token"] for r in rules})

    def __len__(self):
        return len(self.rules)

    def n_tokens(self):
        return len(self.vocab)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return FKGRuleBase(d["vocab"], d["rules"], d.get("edges", []), d.get("meta", {}))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": self.meta, "vocab": self.vocab,
                "rules": self.rules, "edges": self.edges,
            }, f, ensure_ascii=False, indent=2)

    def sample_subset(self, ratio, seed=0):
        """Lấy mẫu ngẫu nhiên ratio% số luật — dùng mô phỏng KB6 khi chưa có
        đủ 5 mức nén thật từ FKGS Chương 2."""
        rng = random.Random(seed)
        n_keep = max(1, int(math.ceil(len(self.rules) * ratio)))
        kept = rng.sample(self.rules, n_keep)
        used_tokens = set()
        for r in kept:
            used_tokens.update(r["antecedent_tokens"])
            used_tokens.add(r["consequent_token"])
        new_vocab = [t for t in self.vocab if t in used_tokens]
        new_edges = build_cooccurrence_edges(kept)
        meta = dict(self.meta)
        meta.update(edge_type_counts(new_edges))
        return FKGRuleBase(new_vocab, kept, new_edges, meta)


def build_cooccurrence_edges(rules):
    from collections import defaultdict

    cooccurrence = defaultdict(int)
    token_count = defaultdict(int)
    for rule in rules:
        tokens = list(dict.fromkeys(rule["antecedent_tokens"]))
        for token in tokens:
            token_count[token] += 1
        for source in tokens:
            for target in tokens:
                if source != target:
                    cooccurrence[(source, target)] += 1

    edges = []
    for (source, target), count in sorted(cooccurrence.items()):
        source_modality = token_modality(source)
        target_modality = token_modality(target)
        if source_modality and target_modality:
            edge_type = ("intra_modal" if source_modality == target_modality
                         else "cross_modal")
        else:
            edge_type = "cooccurrence"
        edges.append({
            "u": source,
            "v": target,
            "mu": round(min(count / max(token_count[source], 1), 1.0), 6),
            "type": edge_type,
        })
    return edges


def token_modality(token):
    return token.split("::", 1)[0] if "::" in token else None


def token_attribute(token):
    if "=" in token:
        return token.rsplit("=", 1)[0]
    return token.rsplit("-", 1)[0]


def edge_type_counts(edges):
    counts = {"intra_modal_edge_count": 0, "cross_modal_edge_count": 0,
              "cooccurrence_edge_count": 0}
    key_by_type = {
        "intra_modal": "intra_modal_edge_count",
        "cross_modal": "cross_modal_edge_count",
        "cooccurrence": "cooccurrence_edge_count",
    }
    for edge in edges:
        counts[key_by_type.get(edge.get("type"), "cooccurrence_edge_count")] += 1
    return counts


def load_test_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["samples"]


def save_test_samples(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)


# ============================================================
# BỘ SINH DỮ LIỆU TỔNG HỢP — dùng để kiểm thử toàn bộ pipeline
# ============================================================
def generate_synthetic_fkg(n_attrs=6, n_labels_per_attr=3, n_rules=200,
                            n_classes=2, seed=42, dataset_name="SYNTH"):
    """Sinh một FKG tổng hợp có cấu trúc THỰC (không random thuần túy):
    mỗi thuộc tính có n_labels_per_attr nhãn ngôn ngữ; luật được sinh sao
    cho tồn tại một mối liên hệ thống kê thật giữa tổ hợp tiền đề và nhãn
    lớp (để accuracy suy luận có ý nghĩa kiểm thử, không phải nhiễu ngẫu
    nhiên hoàn toàn).
    """
    rng = random.Random(seed)
    attrs = [f"A{i}" for i in range(n_attrs)]
    levels = ["Low", "Medium", "High"][:n_labels_per_attr]
    vocab = [f"{a}-{l}" for a in attrs for l in levels]
    class_tokens = [f"class-{c}" for c in range(n_classes)]
    vocab = vocab + class_tokens

    # Quy luật ẩn thật: nếu đa số thuộc tính "High" -> class-1, ngược lại class-0
    # (n_classes=2 mặc định); tổng quát hoá cho n_classes>2 bằng chia đều.
    rules = []
    for rid in range(n_rules):
        n_ante = rng.randint(2, min(4, n_attrs))
        chosen_attrs = rng.sample(attrs, n_ante)
        ante_tokens = []
        high_count = 0
        for a in chosen_attrs:
            lvl = rng.choice(levels)
            if lvl == "High":
                high_count += 1
            ante_tokens.append(f"{a}-{lvl}")
        # Quy luật ẩn + nhiễu 15% để không "quá dễ" (giống dữ liệu thật có nhiễu)
        ratio = high_count / n_ante
        true_class = min(n_classes - 1, int(ratio * n_classes))
        if rng.random() < 0.15:
            true_class = rng.randrange(n_classes)
        conf = round(0.6 + 0.35 * rng.random(), 3)
        supp = round(0.02 + 0.20 * rng.random(), 3)
        rules.append({
            "id": rid,
            "antecedent_tokens": ante_tokens,
            "consequent_token": f"class-{true_class}",
            "confidence": conf,
            "support": supp,
        })

    edges = build_cooccurrence_edges(rules)

    meta = {"dataset": dataset_name, "n_classes": n_classes,
            "class_names": class_tokens}
    return FKGRuleBase(vocab, rules, edges, meta)


def generate_synthetic_test_samples(fkg: FKGRuleBase, n_samples=300, seed=123,
                                     avg_records_per_patient=1.8):
    """Sinh mẫu test có nhãn thật NHẤT QUÁN với quy luật ẩn đã dùng để sinh
    luật ở generate_synthetic_fkg (đếm số thuộc tính 'High' được mờ hoá).

    QUAN TRỌNG: mô phỏng ĐÚNG đặc điểm của BRSET — mỗi bệnh nhân có thể có
    NHIỀU bản ghi (ví dụ ảnh hai mắt, nhiều lần chụp). Các bản ghi của CÙNG
    một bệnh nhân được sinh ra với hồ sơ mờ hoá TƯƠNG QUAN với nhau (không
    độc lập hoàn toàn), mô phỏng thực tế rằng hai ảnh của cùng bệnh nhân
    thường giống nhau hơn hai ảnh của hai bệnh nhân khác nhau. Đây là điểm
    mấu chốt khiến việc chia dữ liệu KHÔNG theo patient_id sẽ gây rò rỉ:
    nếu một bản ghi của bệnh nhân X vào train và bản ghi khác của CHÍNH
    bệnh nhân X vào test, mô hình có thể "nhớ" đặc điểm riêng của X thay vì
    học quy luật tổng quát, khiến accuracy đo được bị phóng đại giả tạo.
    """
    rng = random.Random(seed)
    attrs = sorted({t.rsplit("-", 1)[0] for t in fkg.vocab if not t.startswith("class-")})
    levels = ["Low", "Medium", "High"]
    n_classes = fkg.meta.get("n_classes", 2)
    samples = []
    n_patients = max(1, int(n_samples / avg_records_per_patient))
    sample_id = 0
    for p in range(n_patients):
        patient_id = f"P{p:05d}"
        # Mỗi bệnh nhân có một MỨC ƯA THÍCH riêng cho từng thuộc tính (patient-
        # level bias), khiến các bản ghi của cùng bệnh nhân TƯƠNG QUAN với
        # nhau hơn hai bệnh nhân khác nhau -- đây chính là nguồn rò rỉ nếu
        # chia dữ liệu không theo patient_id. QUAN TRỌNG: công thức dưới đây
        # giữ ĐÚNG tính công bằng cơ bản giữa 3 mức (mỗi mức có xác suất nền
        # ~1/3 như hàm sinh luật gốc dùng), patient_bias chỉ CỘNG THÊM một độ
        # lệch nhẹ có chủ đích lên ĐÚNG MỘT mức ưa thích của bệnh nhân đó
        # (ngẫu nhiên chọn Low/Medium/High), không thiên vị hệ thống về phía
        # "High" như một bản nháp trước đã mắc lỗi (đã kiểm chứng: công thức
        # cũ khiến High thắng dominant 62.5% thay vì ~33% công bằng).
        patient_pref = {a: rng.choice(levels) for a in attrs}
        bias_strength = 0.15  # độ lệch nhẹ, đủ tạo tương quan nhưng không áp đảo
        n_records = 1 + (1 if rng.random() < 0.5 else 0) + (1 if rng.random() < 0.3 else 0)
        for _ in range(n_records):
            if sample_id >= n_samples:
                break
            membership = {}
            high_count = 0
            for a in attrs:
                raw = [rng.random() for _ in levels]
                pref_idx = levels.index(patient_pref[a])
                raw[pref_idx] += bias_strength
                s = sum(raw)
                degs = [x / s for x in raw]
                dominant = levels[degs.index(max(degs))]
                if dominant == "High":
                    high_count += 1
                for lvl, d in zip(levels, degs):
                    tok = f"{a}-{lvl}"
                    if tok in fkg.token2idx:
                        membership[tok] = round(d, 4)
            ratio = high_count / max(1, len(attrs))
            true_class = min(n_classes - 1, int(ratio * n_classes))
            if rng.random() < 0.15:
                true_class = rng.randrange(n_classes)
            samples.append({"membership": membership, "label": f"class-{true_class}",
                             "patient_id": patient_id})
            sample_id += 1
        if sample_id >= n_samples:
            break
    return samples


def load_raw_records(path):
    """Nạp dữ liệu THÔ (chưa mờ hoá) từ file JSON, định dạng:
    {"records": [{"features": {attr: giá_trị_số}, "label": 0/1,
    "patient_id": "..."}, ...]}
    Đây là định dạng ĐẦU VÀO cho RuleMiningPipeline.fit_and_mine()/transform(),
    KHÁC với FKGRuleBase.load() (vốn nạp luật ĐÃ KHAI PHÁ SẴN, không còn
    dùng làm đầu vào chính cho các script thực nghiệm sau khi sửa lỗi
    "FKG cố định không khai phá lại theo fold")."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["records"]


def save_raw_records(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)


def generate_synthetic_prefuzzified_records(n_patients=200, n_attrs=8, n_levels=3,
                                              seed=42, avg_records_per_patient=1.3):
    """Sinh dữ liệu tổng hợp mô phỏng ĐÚNG tình huống BRSET của bạn: mỗi bản
    ghi đã là một luật FRB ĐÃ FUSION (antecedent_tokens mờ hoá sẵn), không
    phải số thô. Dùng để kiểm thử PrefuzzifiedRulePipeline trước khi có dữ
    liệu BRSET thật.
    """
    rng = random.Random(seed)
    attrs = [f"A{i}" for i in range(n_attrs)]
    levels = ["Low", "Medium", "High"][:n_levels]
    records = []
    for p in range(n_patients):
        patient_id = f"P{p:05d}"
        patient_bias = {a: rng.choice(levels) for a in attrs}
        n_records = 1 + (1 if rng.random() < 0.3 else 0)
        for _ in range(n_records):
            tokens = []
            high_count = 0
            for a in attrs:
                if rng.random() < 0.5:
                    lvl = patient_bias[a]
                else:
                    lvl = rng.choice(levels)
                if lvl == "High":
                    high_count += 1
                tokens.append(f"{a}-{lvl}")
            ratio = high_count / n_attrs
            label = 1 if rng.random() < (0.15 + 0.7 * ratio) else 0
            records.append({"antecedent_tokens": tokens, "label": label,
                             "patient_id": patient_id})
    return records


def generate_synthetic_raw_records(n_patients=200, n_attrs=6, seed=42,
                                     avg_records_per_patient=1.5, dataset_name="SYNTH"):
    """Sinh dữ liệu THÔ (giá trị số liên tục, CHƯA mờ hoá), có patient_id,
    mô phỏng đúng tình huống thực tế cần khai phá lại luật theo từng fold:
    mỗi bệnh nhân có thể có nhiều bản ghi (tương quan qua patient_bias),
    Outcome phụ thuộc THẬT vào tổ hợp giá trị các thuộc tính.

    Trả về list[dict]: {"features": {attr: giá_trị_số}, "label": 0/1,
    "patient_id": str} -- ĐÂY LÀ ĐẦU VÀO ĐÚNG cho RuleMiningPipeline.fit_and_mine()/
    transform(), KHÁC với generate_synthetic_test_samples() (vốn sinh sẵn
    membership đã mờ hoá, không dùng được để kiểm thử "khai phá lại theo fold"
    một cách có ý nghĩa).
    """
    rng = random.Random(seed)
    attrs = [f"A{i}" for i in range(n_attrs)]
    records = []
    for p in range(n_patients):
        patient_id = f"P{p:05d}"
        patient_bias = {a: rng.random() for a in attrs}  # thiên hướng riêng mỗi bệnh nhân
        n_records = 1 + (1 if rng.random() < 0.4 else 0)
        for _ in range(n_records):
            features = {}
            score = 0.0
            for a in attrs:
                val = rng.gauss(patient_bias[a] * 10, 2.0)  # giá trị số liên tục
                features[a] = val
                score += val * (1.0 if patient_bias[a] > 0.5 else -0.3)
            prob = 1 / (1 + math.exp(-0.15 * (score - n_attrs * 3)))
            label = 1 if rng.random() < prob else 0
            records.append({"features": features, "label": label, "patient_id": patient_id})
    return records


if __name__ == "__main__":
    # Tự kiểm tra nhanh: sinh + lưu + nạp lại
    fkg = generate_synthetic_fkg()
    print(f"Sinh FKG tổng hợp: {len(fkg)} luật, {fkg.n_tokens()} token, "
          f"{len(fkg.edges)} cạnh")
    samples = generate_synthetic_test_samples(fkg, n_samples=50)
    print(f"Sinh {len(samples)} mẫu test")
    fkg.save("/tmp/test_fkg.json")
    fkg2 = FKGRuleBase.load("/tmp/test_fkg.json")
    assert len(fkg2) == len(fkg), "Lỗi round-trip save/load!"
    print("Round-trip save/load: OK")
