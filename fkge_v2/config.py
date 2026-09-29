"""
config.py — Cấu hình trung tâm cho toàn bộ bộ thực nghiệm FKG-E (KB1-KB6,
Ablation, Baseline), khớp đúng ký hiệu và công thức trong Chương 3 luận án
(Mục 3.4 Mô hình FKG-E, Mục 3.5.4 Kịch bản đánh giá FKG-E).

CHỈNH SỬA DUY NHẤT BẠN CẦN LÀM trước khi chạy trên dữ liệu thật:
  1. PATHS.BRSET_RAW_FILE, DIABETES_KAGGLE_RAW_FILE, HEALTHCARE_DIABETES_RAW_FILE
     -> trỏ tới file DỮ LIỆU THÔ (features số + patient_id + label), CHƯA
     mờ hoá, CHƯA khai phá luật (định dạng: xem load_raw_records() trong
     data/fkg_io.py).
  2. Viết một lớp con kế thừa RuleMiningPipeline (data/pipeline_interface.py)
     gọi ĐÚNG pipeline khai phá luật thật của bạn (GLCM/mã hoá lâm sàng/mờ
     hoá/Wang-Mendel), thay cho RealisticSyntheticPipeline (chỉ minh hoạ).

QUAN TRỌNG: KHÔNG CÒN dùng file "luật đã khai phá sẵn" (rules.json) làm
đầu vào trực tiếp cho các script KB1-KB6/Ablation/Baseline nữa -- luật
LUÔN được khai phá LẠI bên trong từng fold qua pipeline, để tránh rò rỉ
dữ liệu qua bước fitting tham số mờ hoá/khai phá luật (Mục 3.5.1 Chương 3).
"""
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# ============================================================
# ĐƯỜNG DẪN DỮ LIỆU — SỬA THEO MÔI TRƯỜNG THẬT CỦA BẠN
# ============================================================
class PATHS:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    REPOSITORY_ROOT = os.path.dirname(ROOT)
    DATA_DIR = os.path.join(ROOT, "data_real")   # nơi đặt file luật thật
    OUTPUT_DIR = os.path.join(ROOT, "outputs")
    BRSET_FRB_PACKAGE = os.path.join(
        REPOSITORY_ROOT, "Source_code", "data", "exports", "frb_patient_id_20260923")

    # --- Input bắt buộc cho KB1 (3 bộ dữ liệu) ---
    # --- BRSET: luật FRB ĐÃ FUSION ảnh+bảng theo patient_id (đã mờ hoá sẵn,
    # mỗi bản ghi có "antecedent_tokens", KHÔNG phải số thô). Dùng
    # PrefuzzifiedRulePipeline -- xem data/pipeline_interface.py.
    # Định dạng: {"records": [{"antecedent_tokens": ["Age-High",...],
    #             "label": 0/1, "patient_id": "..."}]}
    BRSET_RAW_FILE = os.path.join(DATA_DIR, "brset_raw.json")

    # --- Diabetes-Kaggle/Healthcare: số thô (features: {attr: giá trị số}),
    # CHƯA mờ hoá. Dùng RealisticSyntheticPipeline (tự học ngưỡng phân vị).
    DIABETES_KAGGLE_RAW_FILE = os.path.join(DATA_DIR, "diabetes_kaggle_raw.json")
    HEALTHCARE_DIABETES_RAW_FILE = os.path.join(DATA_DIR, "healthcare_diabetes_raw.json")


# FRB đầu vào chính cho FKG-E phải chứa cả đặc trưng ảnh và bảng. Các modality
# đơn lẻ chỉ dùng cho đối chứng FKG-UM, không đại diện cho FKG-MM.
BRSET_PRIMARY_MODALITY = "fusion"


# ============================================================
# THAM SỐ FKG-E MẶC ĐỊNH (khớp Mục 3.4 luận án)
# ============================================================
class FKGE:
    d = 32                  # chiều nhúng mặc định (KB3 sẽ quét d khác)
    w = None                # đồng xuất hiện trên toàn luật; KB5 giữ w=2 làm đối chứng cũ
    K_neg = 5                # số mẫu âm negative sampling (KB5 quét K khác)
    lam_node = 1.0           # lambda: trọng số L_node
    beta_rule = 1.0          # trọng số L_SGNS; L_rule độc lập chưa triển khai
    # QUAN TRỌNG — ĐÃ HIỆU CHỈNH BẰNG THỰC NGHIỆM (xem ghi chú cuối file):
    # dù loss_and_grad_step() đã CHUẨN HÓA mỗi thành phần theo số phần tử
    # trong batch của nó (mean, không phải sum), gamma_inf=0.5/delta_pred=1.0
    # NGANG BẰNG lam_node/beta_rule=1.0 vẫn KHÔNG đủ để tín hiệu phân loại
    # (Pred/Inf) thắng thế tín hiệu học biểu diễn thuần túy (SGNS/Node) trên
    # dữ liệu tổng hợp dùng để kiểm thử — mô hình sụp về việc luôn đoán lớp
    # đa số (majority-class collapse). Cần delta_pred lớn hơn beta_rule một
    # hệ số đáng kể (thực nghiệm cho thấy ~20x mới đủ) để vượt qua baseline
    # lớp đa số. Coi hai giá trị dưới đây là ĐIỂM KHỞI ĐẦU, KHÔNG PHẢI giá
    # trị đã tối ưu — BẮT BUỘC chạy KB4 (quét lambda, beta) và tự mở rộng
    # quét thêm gamma_inf/delta_pred trên dữ liệu BRSET thật của bạn, kiểm
    # tra accuracy có vượt rõ ràng baseline lớp đa số của TẬP TEST THẬT hay
    # chưa, trước khi tin dùng bất kỳ con số nào cho báo cáo chính thức.
    gamma_inf = 2.0           # gamma: trọng số L_inf (KL với FISA)
    delta_pred = 20.0         # trọng số L_pred (cross-entropy với nhãn thật)
    weight_decay = 1e-5
    aggregation = "class_max"
    max_pairs_per_epoch = 2000  # mẫu SGD đều từ mọi cặp toàn luật ở mỗi epoch
    lr = 0.1                  # đã kiểm chứng: lr thấp hơn (0.01-0.05) khiến
                                # mô hình khó vượt qua baseline lớp đa số
                                # trong ngân sách epochs vừa phải
    epochs = 100               # tăng từ 60 lên 100 cùng lý do với lr ở trên
    batch_size = 64
    tau_softmax = 1.0         # nhiệt độ softmax khi suy luận (Công thức 3.107)
    pooling = "weighted"      # "mean" | "max" | "weighted" (Định nghĩa 3.3)
    alpha_pool = 0.7          # trọng số hệ quả trong weighted pooling
    seed = 42


# ============================================================
# KB3: Độ nhạy theo số chiều nhúng
# ============================================================
KB3_DIMS = [8, 16, 32, 64, 128]

# ============================================================
# KB4: Độ nhạy theo (lambda, beta) — quét lưới
# ============================================================
KB4_LAMBDA_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]
KB4_BETA_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]
KB4_MULTIPLIERS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]

# ============================================================
# KB5: Độ nhạy theo (w, K) skip-gram
# ============================================================
KB5_W_GRID = [None, 2]  # toàn luật (định nghĩa mới) so với cửa sổ w=2 (đối chứng)
KB5_K_GRID = [2, 5, 10]

# ============================================================
# KB6: Khả năng mở rộng quy mô — tỉ lệ mẫu từ FKGS
# ============================================================
KB6_SAMPLE_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]

# ============================================================
# Đánh giá / kiểm định thống kê
# ============================================================
class EVAL:
    N_SEEDS = 5               # số seed lặp lại mỗi cấu hình (giảm phương sai)
    K_FOLD = 5                # k-fold cross-validation
    CONFIDENCE = 0.95
    QUICK = False
