"""
data/kfold_utils.py — Chia dữ liệu theo Stratified Group K-Fold, dùng
patient_id làm biến nhóm, đúng Mục 3.5.1 Chương 3:

    P_train^(k) ∩ P_test^(k) = ∅   với mọi fold k        (Công thức 3.110)

Nghĩa là: mọi bản ghi của CÙNG một bệnh nhân chỉ được xuất hiện ở MỘT
phía (train HOẶC test) trong mỗi fold — không bao giờ cả hai.
"""
import numpy as np
from collections import Counter

try:
    from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
    _HAS_SKLEARN_GROUP = True
except ImportError:
    _HAS_SKLEARN_GROUP = False


def make_patient_kfold(samples, k=5, seed=42, stratify=True):
    """Trả về list gồm k tuple (train_idx, test_idx) — chỉ số vào `samples`.

    Nếu thiếu 'patient_id' ở bất kỳ mẫu nào, DỪNG NGAY và báo lỗi rõ ràng
    thay vì âm thầm coi mỗi mẫu là một 'bệnh nhân' riêng (điều đó sẽ vô
    hiệu hoá hoàn toàn mục đích chống rò rỉ dữ liệu).
    """
    missing = [i for i, s in enumerate(samples) if "patient_id" not in s or not s["patient_id"]]
    if missing:
        raise ValueError(
            f"CÓ {len(missing)}/{len(samples)} MẪU THIẾU 'patient_id'! "
            f"Không thể chia k-fold an toàn theo bệnh nhân. Kiểm tra lại file "
            f"dữ liệu đầu vào — mọi mẫu PHẢI có trường 'patient_id' (xem "
            f"README.md mục 3.2 và data/fkg_io.py)."
        )

    groups = np.array([s["patient_id"] for s in samples])
    y = np.array([s["label"] for s in samples])
    n_unique_patients = len(set(groups))
    if n_unique_patients < k:
        raise ValueError(
            f"Chỉ có {n_unique_patients} bệnh nhân duy nhất, không đủ để chia "
            f"{k}-fold theo nhóm (cần ít nhất {k} bệnh nhân, lý tưởng nhiều hơn "
            f"đáng kể để mỗi fold có đủ đa dạng)."
        )

    X_dummy = np.zeros((len(samples), 1))
    if stratify and _HAS_SKLEARN_GROUP:
        try:
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            folds = list(splitter.split(X_dummy, y, groups))
        except ValueError as e:
            print(f"  !! StratifiedGroupKFold thất bại ({e}), chuyển sang GroupKFold "
                  f"thường (không đảm bảo cân bằng nhãn giữa các fold).")
            splitter = GroupKFold(n_splits=k)
            folds = list(splitter.split(X_dummy, y, groups))
    elif _HAS_SKLEARN_GROUP:
        splitter = GroupKFold(n_splits=k)
        folds = list(splitter.split(X_dummy, y, groups))
    else:
        folds = _manual_group_kfold(groups, k, seed)

    _verify_no_patient_leakage(samples, folds)
    return folds


def _manual_group_kfold(groups, k, seed):
    """Cài đặt GroupKFold thủ công (dự phòng nếu sklearn không có sẵn):
    gán mỗi bệnh nhân duy nhất vào một fold theo round-robin sau khi xáo
    trộn ngẫu nhiên, bảo đảm không có bệnh nhân nào xuất hiện ở 2 fold."""
    rng = np.random.RandomState(seed)
    unique_patients = list(sorted(set(groups)))
    rng.shuffle(unique_patients)
    patient_to_fold = {p: i % k for i, p in enumerate(unique_patients)}
    fold_of_sample = np.array([patient_to_fold[g] for g in groups])
    folds = []
    for fold_id in range(k):
        test_idx = np.where(fold_of_sample == fold_id)[0]
        train_idx = np.where(fold_of_sample != fold_id)[0]
        folds.append((train_idx, test_idx))
    return folds


def _verify_no_patient_leakage(samples, folds):
    """Kiểm tra BẮT BUỘC sau khi chia: không bệnh nhân nào xuất hiện ở cả
    train và test của cùng một fold. Đây là bài kiểm thử tự động, chạy mỗi
    lần chia dữ liệu — nếu phát hiện rò rỉ, dừng chương trình ngay lập tức
    thay vì để người dùng vô tình dùng kết quả sai."""
    groups = np.array([s["patient_id"] for s in samples])
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train_patients = set(groups[train_idx])
        test_patients = set(groups[test_idx])
        overlap = train_patients & test_patients
        assert not overlap, (
            f"RÒ RỈ DỮ LIỆU tại fold {fold_id}: {len(overlap)} bệnh nhân xuất "
            f"hiện ở CẢ train và test ({list(overlap)[:5]}...). Đây là lỗi "
            f"nghiêm trọng vi phạm trực tiếp Công thức (3.110) Chương 3 — "
            f"DỪNG, không sử dụng kết quả cho đến khi sửa xong."
        )


def get_train_test_single_fold(samples, k=5, seed=42, fold_id=0):
    """Lấy (train, test) từ ĐÚNG MỘT fold (mặc định fold 0) của Stratified
    Group K-Fold theo patient_id. Dùng cho các thực nghiệm QUÉT LƯỚI siêu
    tham số (KB3, KB4, KB5) — nơi số tổ hợp cần thử quá lớn để lặp đủ cả
    k-fold cho mỗi tổ hợp trong thời gian hợp lý (đúng tinh thần "inner
    loop" của nested cross-validation, Mục 3.5.2 Chương 3: dùng 1 phần chia
    nhanh để dò xu hướng siêu tham số, rồi mới đánh giá cấu hình tốt nhất
    bằng ĐẦY ĐỦ k-fold ở KB1/KB2). VẪN bảo đảm không rò rỉ patient_id giữa
    train/test của fold được chọn.
    """
    folds = make_patient_kfold(samples, k=k, seed=seed)
    train_idx, test_idx = folds[fold_id]
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    return train, test


def summarize_folds(samples, folds):
    """In tóm tắt số bệnh nhân/mẫu mỗi fold + phân bố nhãn, để kiểm tra
    trực quan trước khi chạy thực nghiệm dài."""
    groups = np.array([s["patient_id"] for s in samples])
    labels = np.array([s["label"] for s in samples])
    print(f"  Tổng số mẫu: {len(samples)}, số bệnh nhân duy nhất: {len(set(groups))}")
    for i, (train_idx, test_idx) in enumerate(folds):
        n_train_p = len(set(groups[train_idx]))
        n_test_p = len(set(groups[test_idx]))
        test_label_dist = Counter(labels[test_idx])
        print(f"  Fold {i}: train={len(train_idx)} mẫu/{n_train_p} bệnh nhân | "
              f"test={len(test_idx)} mẫu/{n_test_p} bệnh nhân | "
              f"phân bố nhãn test={dict(test_label_dist)}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples

    fkg = generate_synthetic_fkg(n_rules=200, seed=1)
    samples = generate_synthetic_test_samples(fkg, n_samples=300, seed=2)

    n_unique = len(set(s["patient_id"] for s in samples))
    print(f"Sinh {len(samples)} mẫu từ {n_unique} bệnh nhân "
          f"(trung bình {len(samples)/n_unique:.2f} bản ghi/bệnh nhân)")

    folds = make_patient_kfold(samples, k=5, seed=42)
    summarize_folds(samples, folds)
    print("\nKiểm thử: không phát hiện rò rỉ dữ liệu qua _verify_no_patient_leakage -- OK")
