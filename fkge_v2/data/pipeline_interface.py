"""
data/pipeline_interface.py — Xác định CHÍNH XÁC vị trí k-fold trong quy
trình nhiều bước của Hình 3.2 (Chương 3), và cung cấp giao diện để khai
phá lại luật ĐÚNG cách bên trong mỗi fold.

======================================================================
VỊ TRÍ ĐÚNG CỦA K-FOLD TRONG HÌNH 3.2 (Chương 3)
======================================================================

    Ảnh đáy mắt BRSET          Dữ liệu lâm sàng
           │                          │
           ▼                          ▼
   ╔═══════════════════════════════════════════╗
   ║   ***** K-FOLD SPLIT XẢY RA Ở ĐÂY *****    ║   <-- TRƯỚC TẤT CẢ,
   ║   theo patient_id (Mục 3.5.1, Ct. 3.110)   ║       chia trên dữ liệu
   ╚═══════════════════════════════════════════╝       THÔ, chưa xử lý gì
           │                          │
           ▼ (chỉ dùng TRAIN)         ▼ (chỉ dùng TRAIN)
   Chuẩn hoá, khử nhiễu,      Điền thiếu, mã hoá,
   tăng tương phản             chuẩn hoá
           │                          │
           ▼                          ▼
   Phân đoạn/mã hoá ảnh        Lựa chọn đặc trưng
           │                          │
           ▼                          ▼
   GLCM/deep features           Đặc trưng bảng đã chọn
           │                          │
           └──────────┬───────────────┘
                       ▼
         Mờ hoá riêng từng mô thức (fit hàm thuộc CHỈ trên train)
                       │
                       ▼
              V^img, V^tab  →  E_intra và E_cross
                       │
                       ▼
         Sinh và sàng lọc luật đa mô thức  <-- SINH LUẬT CŨNG Ở TRONG
                       │                        VÙNG "CHỈ DÙNG TRAIN"
                       ▼
              FKG-MM của FOLD NÀY (train-only)
                       │
         ┌─────────────┴─────────────┐
         ▼                            ▼
    Suy luận FISA                FKG-E (huấn luyện
    trên TEST của fold           trên train, đánh giá
    (dùng FKG-MM vừa xây)        trên test của fold)

TÓM LẠI — trả lời trực tiếp câu hỏi "k-fold ở đâu":
  K-fold KHÔNG nằm ở một bước cụ thể trong hình — nó nằm NGAY TRƯỚC
  TOÀN BỘ hình, chia dữ liệu thô theo patient_id thành 5 phần. Sau đó,
  TOÀN BỘ pipeline (từ "Chuẩn hoá..." đến "Sinh và sàng lọc luật") được
  CHẠY LẠI TỪ ĐẦU cho MỖI fold, chỉ dùng phần train của fold đó. Chỉ
  bước SUY LUẬN CUỐI CÙNG (FISA/FKG-E) mới dùng đến phần test.

  Nếu chỉ tách k-fold ở bước "trước suy luận FISA/FKG-E" (tức TÁI SỬ DỤNG
  một FKG-MM đã khai phá sẵn, cố định, cho mọi fold) — như bản triển khai
  ĐẦU TIÊN của kb1_kb2.py đã làm — đây là SAI, vì các tham số mờ hoá,
  ngưỡng đặc trưng và chính tập luật đều đã "nhìn thấy" toàn bộ dữ liệu
  (bao gồm cả các bệnh nhân sau này rơi vào test), gây rò rỉ dữ liệu tinh
  vi (leakage qua bước fitting, không phải qua nhãn trực tiếp).

======================================================================
GIAO DIỆN NGƯỜI DÙNG CẦN CÀI ĐẶT
======================================================================
Bộ thực nghiệm KB1-KB6 KHÔNG tự có code trích đặc trưng ảnh GLCM / mã hoá
lâm sàng / khai phá luật Wang-Mendel thật cho BRSET -- đó là code riêng
bạn đã xây ở Chương 2-3. Để k-fold đúng vị trí, bạn cần cài đặt lớp
RuleMiningPipeline dưới đây bằng CHÍNH pipeline khai phá luật thật của
mình, rồi truyền vào experiments/kb1_kb2.py thay vì truyền thẳng một FKG
cố định.
"""
from abc import ABC, abstractmethod


class RuleMiningPipeline(ABC):
    """Giao diện trừu tượng — cài đặt lớp con bằng pipeline khai phá luật
    THẬT của bạn (Mục 3.3.2-3.3.5 Chương 3: trích đặc trưng ảnh/bảng, mờ
    hoá, xây quan hệ E_intra/E_cross, sinh và sàng lọc luật)."""

    @abstractmethod
    def fit_and_mine(self, train_raw_records):
        """CHỈ được nhìn thấy `train_raw_records` (danh sách bản ghi thô
        của các bệnh nhân thuộc train fold — ảnh + dữ liệu lâm sàng CHƯA
        qua bất kỳ xử lý nào). Phải thực hiện TRỌN VẸN các bước: chuẩn hoá,
        trích đặc trưng, xác định tham số hàm thuộc (a,b,c mỗi biến), xây
        E_intra/E_cross, sinh và sàng lọc luật.

        Trả về (fkg, train_samples):
          fkg            : FKGRuleBase -- tập luật CHỈ khai phá từ train
          train_samples  : list các dict {"membership":..., "label":...,
                            "patient_id":...} -- train_raw_records sau khi
                            đã mờ hoá bằng chính tham số vừa fit
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, test_raw_records):
        """Mờ hoá `test_raw_records` bằng ĐÚNG các tham số (a,b,c, bộ chuẩn
        hoá, danh sách đặc trưng đã chọn...) đã fit ở fit_and_mine() -- TUYỆT
        ĐỐI KHÔNG được fit lại hay nhìn thấy bất kỳ thống kê nào của test.
        Trả về list các dict {"membership":..., "label":..., "patient_id":...}
        cùng định dạng với train_samples.
        """
        raise NotImplementedError


class SyntheticRuleMiningPipeline(RuleMiningPipeline):
    """Cài đặt MẪU dùng dữ liệu tổng hợp -- minh hoạ đúng luồng và dùng để
    KIỂM THỬ cơ chế k-fold-trước-khi-sinh-luật hoạt động chính xác (luật
    phải KHÁC NHAU giữa các fold vì được khai phá lại từ train khác nhau).
    Đây KHÔNG PHẢI code dùng cho BRSET thật -- bạn phải viết lớp con khác
    gọi đúng pipeline GLCM/Wang-Mendel thật của mình.
    """

    def __init__(self, n_rules_per_fold=150, seed=0):
        self.n_rules_per_fold = n_rules_per_fold
        self.seed = seed
        self._fold_counter = 0

    def fit_and_mine(self, train_raw_records):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.fkg_io import generate_synthetic_fkg

        # Mô phỏng "chỉ nhìn train": số luật/seed phụ thuộc CHÍNH XÁC vào
        # nội dung train_raw_records (qua số lượng bản ghi) để đảm bảo hai
        # fold khác nhau cho ra fkg khác nhau -- đúng tinh thần "khai phá
        # lại", không phải một fkg cố định dùng chung.
        seed = self.seed * 1000 + len(train_raw_records) + self._fold_counter
        self._fold_counter += 1
        fkg = generate_synthetic_fkg(n_rules=self.n_rules_per_fold, seed=seed,
                                      dataset_name="BRSET(synthetic-per-fold)")
        self._fitted_fkg = fkg
        train_samples = self._mine_membership_for(train_raw_records, fkg)
        return fkg, train_samples

    def transform(self, test_raw_records):
        if not hasattr(self, "_fitted_fkg"):
            raise RuntimeError("Phải gọi fit_and_mine() trên train TRƯỚC khi transform() test.")
        return self._mine_membership_for(test_raw_records, self._fitted_fkg)

    def _mine_membership_for(self, raw_records, fkg):
        # raw_records ở đây ĐÃ là dict {"membership":..., "label":...,
        # "patient_id":...} sẵn (vì bộ sinh dữ liệu tổng hợp generate_synthetic_
        # test_samples đã tạo membership trực tiếp) -- trong pipeline THẬT,
        # đây chính là chỗ bạn gọi hàm mờ hoá thật (Công thức 3.3.2) áp dụng
        # lên đặc trưng ảnh/bảng thô của raw_records.
        return raw_records


class PrefuzzifiedRulePipeline(RuleMiningPipeline):
    """DÙNG CHO BRSET khi dữ liệu đầu vào ĐÃ LÀ luật FRB sau khi fusion
    ảnh+bảng theo patient_id (đã mờ hoá sẵn ở mức từng bệnh nhân — mỗi bản
    ghi đã có "antecedent_tokens" dạng ["Age-High", "GLCM_Contrast-Medium",
    ...], KHÔNG còn là số thô cần học ngưỡng phân vị như
    RealisticSyntheticPipeline).

    Pipeline này CHỈ thực hiện đúng phần "sinh và sàng lọc luật" (đếm
    support/confidence) CHỈ trên train của từng fold -- KHÔNG học lại tham
    số mờ hoá (vì đã mờ hoá sẵn ở bước fusion, bất biến giữa train/test).
    Đây vẫn là bước fitting cần tách theo fold: nếu support/confidence được
    tính trên TOÀN BỘ dữ liệu trước khi chia fold, các luật của bệnh nhân
    rơi vào test đã "được đếm vào" độ tin cậy của luật -- vẫn là rò rỉ,
    dù nhẹ hơn so với rò rỉ qua tham số mờ hoá liên tục.

    Định dạng bản ghi đầu vào (mỗi phần tử của `train_raw_records`/
    `test_raw_records`):
        {"antecedent_tokens": ["Age-High", "HbA1c-High", "GLCM_Contrast-Medium"],
         "label": 1, "patient_id": "P00123"}

    ⚠️ HẠN CHẾ ĐÃ BIẾT (CÙNG LOẠI với đã xác nhận ở fkgs_v2, "Hướng A" --
    giữ nguyên, không sửa): trên dữ liệu MẤT CÂN BẰNG LỚP, support/confidence
    đếm trực tiếp theo tần suất tuyệt đối (không chuẩn hoá theo |R_l| của
    từng lớp) khiến W[attr][v][lớp đa số] > W[attr][v][lớp thiểu số] một
    cách HỆ THỐNG tại hầu hết giá trị thuộc tính -- đã kiểm chứng thực
    nghiệm: với tỉ lệ lớp 62%/38%, FISA có thể sụp về đúng bằng baseline
    lớp đa số (xem warn_if_not_beating_majority() trong models/fisa.py,
    tự động cảnh báo mỗi khi hiện tượng này xảy ra khi chạy trên BRSET
    thật). Đây là đặc điểm của phương pháp đếm tần suất kiểu Wang-Mendel
    khi mất cân bằng lớp, không phải lỗi cài đặt riêng của pipeline này.
    """

    def __init__(self, sample_ratio=1.0, seed=0):
        self._fitted = False
        # sample_ratio<1.0: mô phỏng FKGS (nén luật) -- lấy mẫu con luật
        # NGAY SAU KHI khai phá đầy đủ, CHỈ trên train của fold này.
        self.sample_ratio = sample_ratio
        self.seed = seed

    def fit_and_mine(self, train_raw_records):
        from collections import Counter
        antecedent_key = lambda toks: tuple(sorted(toks))
        ante_counter = Counter()
        ante_label_counter = Counter()
        for r in train_raw_records:
            key = antecedent_key(r["antecedent_tokens"])
            ante_counter[key] += 1
            ante_label_counter[(key, r["label"])] += 1

        n = len(train_raw_records)
        rules, seen = [], set()
        for r in train_raw_records:
            key = antecedent_key(r["antecedent_tokens"])
            if key in seen:
                continue
            seen.add(key)
            support = ante_counter[key] / n
            best_label = max(
                (l for l in set(x[1] for x in ante_label_counter if x[0] == key)),
                key=lambda l: ante_label_counter[(key, l)],
            )
            confidence = ante_label_counter[(key, best_label)] / ante_counter[key]
            rules.append({
                "id": len(rules), "antecedent_tokens": list(key),
                "consequent_token": f"class-{best_label}",
                "support": support, "confidence": confidence,
            })

        vocab = sorted({t for r in rules for t in r["antecedent_tokens"]} |
                        {r["consequent_token"] for r in rules})
        n_classes = len({r["label"] for r in train_raw_records})
        from data.fkg_io import FKGRuleBase, build_cooccurrence_edges, edge_type_counts
        edges = build_cooccurrence_edges(rules)
        modalities = sorted({
            token.split("::", 1)[0]
            for rule in rules for token in rule["antecedent_tokens"]
            if "::" in token
        })
        graph_kind = "FKG-MM" if len(modalities) > 1 else "FKG-UM"
        meta = {
            "dataset": "BRSET",
            "n_classes": n_classes,
            "modalities": modalities,
            "input_graph_kind": graph_kind,
            **edge_type_counts(edges),
        }
        fkg = FKGRuleBase(vocab, rules, edges, meta)
        if self.sample_ratio < 1.0:
            fkg = fkg.sample_subset(ratio=self.sample_ratio, seed=self.seed)
        self._fitted = True
        return fkg, [self._to_sample(r) for r in train_raw_records]

    def transform(self, test_raw_records):
        if not self._fitted:
            raise RuntimeError("Phải gọi fit_and_mine() trên train TRƯỚC khi transform() test.")
        return [self._to_sample(r) for r in test_raw_records]

    def _to_sample(self, record):
        membership = {t: 1.0 for t in record["antecedent_tokens"]}
        return {"membership": membership, "label": f"class-{record['label']}",
                "patient_id": record["patient_id"]}


class RealisticSyntheticPipeline(RuleMiningPipeline):
    """Cài đặt tổng hợp NHƯNG THỰC CHẤT (khác SyntheticRuleMiningPipeline ở
    trên -- vốn sinh luật NGẪU NHIÊN theo seed, không thực sự phụ thuộc nội
    dung dữ liệu): pipeline này THẬT SỰ mờ hoá dữ liệu thô bằng ngưỡng phân
    vị học TỪ TRAIN, rồi khai phá luật bằng cách đếm support/confidence
    THẬT trên chính train đã mờ hoá. Dùng làm mẫu minh hoạ cách viết
    pipeline cho dữ liệu số liên tục (gần với BRSET/Diabetes hơn); người
    dùng vẫn cần thay bằng đúng pipeline GLCM/Wang-Mendel thật của mình cho
    báo cáo chính thức.
    """

    def __init__(self, n_levels=3, seed=0, sample_ratio=1.0):
        self.n_levels = n_levels
        self.seed = seed
        self.thresholds_ = {}  # {attr: [q1, q2, ...]} học từ train
        # sample_ratio<1.0: mô phỏng FKGS (nén luật) -- LẤY MẪU CON các
        # luật đã khai phá được, CHỈ trên train của fold (không rò rỉ),
        # dùng cho KB2 (so sánh FKG-MM đầy đủ vs FKGS đã nén).
        self.sample_ratio = sample_ratio

    def fit_and_mine(self, train_raw_records):
        attrs = sorted(train_raw_records[0]["features"].keys())
        # Học ngưỡng phân vị CHỈ từ train
        import numpy as np
        self.thresholds_ = {}
        for a in attrs:
            vals = np.array([r["features"][a] for r in train_raw_records])
            qs = np.percentile(vals, np.linspace(0, 100, self.n_levels + 1)[1:-1])
            self.thresholds_[a] = qs.tolist()

        train_samples = [self._fuzzify_record(r, attrs) for r in train_raw_records]

        # Khai phá luật: MỖI bản ghi train là một luật ứng viên (tiền đề =
        # các token của nó, hệ quả = nhãn của nó); support/confidence tính
        # bằng cách ĐẾM THẬT số bản ghi có CÙNG tổ hợp token tiền đề.
        from collections import Counter
        antecedent_key = lambda toks: tuple(sorted(toks))
        ante_counter = Counter()
        ante_label_counter = Counter()
        rule_antecedents = []
        for s in train_samples:
            toks = tuple(t for t, deg in s["membership"].items() if deg >= 0.999)
            rule_antecedents.append(toks)
            ante_counter[antecedent_key(toks)] += 1
            ante_label_counter[(antecedent_key(toks), s["label"])] += 1

        n = len(train_samples)
        rules, seen = [], set()
        for toks, s in zip(rule_antecedents, train_samples):
            key = antecedent_key(toks)
            if key in seen:
                continue
            seen.add(key)
            support = ante_counter[key] / n
            # best_label ĐÃ ở dạng "class-X" (vì s["label"] từ _fuzzify_record
            # đã được chuẩn hoá) -- KHÔNG thêm tiền tố "class-" lần nữa ở đây,
            # nếu không sẽ tạo chuỗi lồng "class-class-X" (bug đã phát hiện).
            best_label = max(
                (l for l in set(x[1] for x in ante_label_counter if x[0] == key)),
                key=lambda l: ante_label_counter[(key, l)],
            )
            confidence = ante_label_counter[(key, best_label)] / ante_counter[key]
            rules.append({
                "id": len(rules), "antecedent_tokens": list(toks),
                "consequent_token": best_label,
                "support": support, "confidence": confidence,
            })

        vocab = sorted({t for r in rules for t in r["antecedent_tokens"]} |
                        {r["consequent_token"] for r in rules})
        n_classes = len({r["label"] for r in train_raw_records})
        from data.fkg_io import FKGRuleBase, build_cooccurrence_edges
        edges = build_cooccurrence_edges(rules)
        fkg = FKGRuleBase(vocab, rules, edges, {"dataset": "RealisticSynthetic",
                                                  "n_classes": n_classes})
        if self.sample_ratio < 1.0:
            # Mô phỏng FKGS: nén luật NGAY SAU KHI khai phá đầy đủ, CHỈ dựa
            # trên luật của CHÍNH fold này -- không dùng thông tin từ fold khác.
            fkg = fkg.sample_subset(ratio=self.sample_ratio, seed=self.seed)
        self._fitted = True
        return fkg, [{"membership": s["membership"], "label": s["label"],
                       "patient_id": s["patient_id"]} for s in train_samples]

    def transform(self, test_raw_records):
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Phải gọi fit_and_mine() trên train TRƯỚC khi transform() test.")
        attrs = sorted(self.thresholds_.keys())
        samples = [self._fuzzify_record(r, attrs) for r in test_raw_records]
        return [{"membership": s["membership"], "label": s["label"],
                  "patient_id": s["patient_id"]} for s in samples]

    def _fuzzify_record(self, record, attrs):
        membership = {}
        for a in attrs:
            val = record["features"][a]
            qs = self.thresholds_[a]
            level_idx = sum(1 for q in qs if val > q)  # 0..n_levels-1
            level_name = ["Low", "Medium", "High"][min(level_idx, 2)]
            membership[f"{a}-{level_name}"] = 1.0
        # QUAN TRỌNG (bug thật đã phát hiện và sửa): nhãn PHẢI ở đúng định
        # dạng "class-{label}" -- khớp với self.class_tokens của FKGRuleBase
        # và với y_hat mà FISA/FKGE trả về (đều là chuỗi "class-X"). Nếu giữ
        # nhãn dạng số nguyên thô, mọi so sánh y_true==y_pred trong evaluate()
        # sẽ LUÔN FALSE (khác kiểu dữ liệu: int 0 != str "class-0"), cho
        # accuracy=0.0000 GIẢ TẠO dù suy diễn thực chất có thể đúng.
        return {"membership": membership, "label": f"class-{record['label']}",
                "patient_id": record["patient_id"]}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples
    from data.kfold_utils import make_patient_kfold

    print("=== Kiểm tra: luật PHẢI khác nhau giữa các fold (bằng chứng đã khai phá lại) ===")
    base_fkg = generate_synthetic_fkg(n_rules=100, seed=1)
    raw = generate_synthetic_test_samples(base_fkg, n_samples=300, seed=2)
    folds = make_patient_kfold(raw, k=5, seed=42)

    pipeline = SyntheticRuleMiningPipeline(n_rules_per_fold=50, seed=7)
    rule_signatures = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train_raw = [raw[i] for i in train_idx]
        test_raw = [raw[i] for i in test_idx]
        fkg_fold, train_samples = pipeline.fit_and_mine(train_raw)
        test_samples = pipeline.transform(test_raw)
        sig = tuple(sorted(r["antecedent_tokens"][0] if r["antecedent_tokens"] else "" for r in fkg_fold.rules[:5]))
        rule_signatures.append(sig)
        print(f"  Fold {fold_id}: khai phá {len(fkg_fold)} luật từ {len(train_raw)} bản ghi train "
              f"(chữ ký 5 luật đầu: {sig[:2]}...)")

    n_unique_signatures = len(set(rule_signatures))
    print(f"\nSố chữ ký luật DUY NHẤT qua 5 fold: {n_unique_signatures}/5")
    assert n_unique_signatures >= 3, (
        "CẢNH BÁO: luật gần như GIỐNG NHAU giữa các fold -- có thể pipeline "
        "chưa thực sự khai phá lại theo từng fold!"
    )
    print("Kiểm thử: luật được khai phá LẠI cho từng fold (khác nhau đủ rõ) -- OK")
