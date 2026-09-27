"""
models/kge_pykeen.py — Chạy TransE/DistMult/ComplEx/RotatE BẰNG THƯ VIỆN
CHUẨN PyKEEN (https://github.com/pykeen/pykeen), thay cho các bản "-lite"
tự viết trong models/kge_baselines.py.

VÌ SAO NÊN DÙNG PYKEEN CHO BÁO CÁO CHÍNH THỨC:
  - Cài đặt đã qua kiểm chứng, được cộng đồng nghiên cứu KGE dùng rộng rãi
    và trích dẫn được (Ali et al., "PyKEEN 1.0", JMLR 2021).
  - Có early stopping, learning-rate scheduler, negative sampling chuẩn
    (theo đúng bài báo gốc của từng mô hình), tránh các đơn giản hoá trong
    bản "-lite" (ví dụ TransELite/DistMultLite ở kge_baselines.py chỉ cập
    nhật 2000 triple ngẫu nhiên/epoch, không có early stopping).
  - Hội đồng phản biện dễ chấp nhận hơn khi baseline dùng implementation
    chuẩn thay vì code tự viết từ đầu.

CÀI ĐẶT (nặng, cần ~2-3GB dung lượng trống do kéo theo PyTorch):
    pip install pykeen

Nếu KHÔNG cài được (ví dụ máy hạn chế dung lượng), dùng
models/kge_baselines.py (Node2VecLite/TransELite/DistMultLite) làm
phương án thay thế nhẹ hơn — đã được ghi rõ trong luận án là baseline
rút gọn ("-lite"), không thay thế hoàn toàn bản chuẩn.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline as pykeen_pipeline
    _HAS_PYKEEN = True
except ImportError:
    _HAS_PYKEEN = False

from data.fkg_to_kg import fkg_to_triples
import numpy as np


PYKEEN_MODEL_NAMES = {
    "TransE": "TransE",
    "DistMult": "DistMult",
    "ComplEx": "ComplEx",
    "RotatE": "RotatE",
}


class PyKeenKGEWrapper:
    """Bọc ngoài PyKEEN để có cùng giao diện fit()/E (bảng nhúng thực thể)
    như các baseline -lite, cho phép tái sử dụng KNNOnEmbedding sẵn có
    trong models/kge_baselines.py mà không cần viết lại downstream classifier.
    """

    def __init__(self, fkg, model_name="TransE", strategy="reified", d=32,
                 epochs=50, seed=42, edge_threshold=0.5, num_negs_per_pos=1):
        if not _HAS_PYKEEN:
            raise ImportError(
                "Chưa cài PyKEEN. Chạy: pip install pykeen  "
                "(cần ~2-3GB dung lượng trống do kéo theo PyTorch). "
                "Nếu không thể cài, dùng models/kge_baselines.py thay thế."
            )
        if model_name not in PYKEEN_MODEL_NAMES:
            raise ValueError(f"model_name phải thuộc {list(PYKEEN_MODEL_NAMES)}")
        self.fkg = fkg
        self.model_name = model_name
        self.strategy = strategy
        self.d = d
        self.epochs = epochs
        self.seed = seed
        self.token2idx = fkg.token2idx  # để tương thích giao diện KNNOnEmbedding

        triples = fkg_to_triples(fkg, strategy=strategy, edge_threshold=edge_threshold)
        if len(triples) == 0:
            raise ValueError("Không sinh được triple nào từ FKG -- kiểm tra lại dữ liệu luật/cạnh.")
        self._triples_array = np.array(triples, dtype=str)
        self.num_negs_per_pos = num_negs_per_pos

    def fit(self):
        t0 = time.time()
        tf = TriplesFactory.from_labeled_triples(self._triples_array)
        training, testing = tf.split([0.9, 0.1], random_state=self.seed)

        result = pykeen_pipeline(
            training=training,
            testing=testing,
            model=PYKEEN_MODEL_NAMES[self.model_name],
            model_kwargs=dict(embedding_dim=self.d),
            training_kwargs=dict(num_epochs=self.epochs, use_tqdm=False),
            negative_sampler_kwargs=dict(num_negs_per_pos=self.num_negs_per_pos),
            random_seed=self.seed,
            device="cpu",
        )
        self.result = result
        self.train_time_s = time.time() - t0

        # Trích bảng nhúng thực thể ra dạng numpy, ÁNH XẠ đúng theo thứ tự
        # fkg.vocab để tương thích với KNNOnEmbedding (vốn lập chỉ mục theo
        # fkg.token2idx). Với chiến lược "reified", các thực thể rule_k KHÔNG
        # nằm trong fkg.vocab -- chỉ trích embedding cho các token có mặt cả
        # trong fkg.vocab lẫn trong tf.entity_to_id.
        #
        # QUAN TRỌNG (đã phát hiện và sửa bug thật): ComplEx dùng embedding
        # SỐ PHỨC (Mệnh đề 3.2.17, Chương 3 -- chính phần ảo là lý do ComplEx
        # biểu diễn được quan hệ bất đối xứng). Nếu ép thẳng sang mảng thực,
        # NumPy/PyTorch tự động CẮT BỎ phần ảo (đã quan sát ComplexWarning
        # thật khi chạy thử) và làm ComplEx suy biến gần như DistMult -- mất
        # đúng phần cấu trúc quan trọng nhất của mô hình. Cách sửa đúng: nối
        # [phần thực, phần ảo] thành một vector thực có số chiều gấp đôi.
        entity_embeddings = result.model.entity_representations[0](indices=None).detach().cpu().numpy()
        if np.iscomplexobj(entity_embeddings):
            entity_embeddings = np.concatenate(
                [entity_embeddings.real, entity_embeddings.imag], axis=1)
            effective_dim = entity_embeddings.shape[1]
        else:
            effective_dim = self.d
        self.E = np.zeros((len(self.fkg.vocab), effective_dim))
        n_found = 0
        for tok, idx in self.token2idx.items():
            if tok in tf.entity_to_id:
                self.E[idx] = entity_embeddings[tf.entity_to_id[tok]]
                n_found += 1
        if n_found == 0:
            raise RuntimeError(
                "Không ánh xạ được BẤT KỲ token nào từ fkg.vocab sang không "
                "gian nhúng PyKEEN -- kiểm tra lại chiến lược chuyển đổi."
            )
        self.n_tokens_embedded = n_found
        return self

    def evaluate_link_prediction(self):
        """Trả về các độ đo chuẩn của KGE (MRR, Hits@K) trên tập test nội
        bộ đã tách từ chính quá trình chuyển đổi triples -- đây là độ đo
        'đúng bài' cho KGE (dự đoán liên kết), KHÁC với accuracy phân loại
        DR/No-DR (đo qua KNNOnEmbedding ở downstream)."""
        metrics = self.result.metric_results.to_dict()
        return metrics


if __name__ == "__main__":
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples
    from models.kge_baselines import KNNOnEmbedding

    if not _HAS_PYKEEN:
        print("PyKEEN chưa được cài -- chạy: pip install pykeen")
        sys.exit(1)

    fkg = generate_synthetic_fkg(n_rules=150, seed=1)
    test_samples = generate_synthetic_test_samples(fkg, n_samples=100, seed=20)

    for model_name in ["TransE", "DistMult"]:
        print(f"\n=== PyKEEN {model_name} (strategy=reified) ===")
        wrapper = PyKeenKGEWrapper(fkg, model_name=model_name, strategy="reified",
                                    d=16, epochs=20, seed=1)
        wrapper.fit()
        print(f"Đã huấn luyện trong {wrapper.train_time_s:.2f}s, "
              f"ánh xạ được {wrapper.n_tokens_embedded}/{len(fkg.vocab)} token")

        clf = KNNOnEmbedding(fkg, wrapper.E, fkg.token2idx, k=5)
        res = clf.evaluate(test_samples)
        print(f"Downstream KNN: accuracy={res['accuracy']:.4f}  f1={res['f1_macro']:.4f}")

        link_pred_metrics = wrapper.evaluate_link_prediction()
        mrr = link_pred_metrics.get("both", {}).get("realistic", {}).get("inverse_harmonic_mean_rank")
        print(f"MRR (link prediction nội bộ, để tham khảo): {mrr}")

    print("\nKiểm thử PyKEEN wrapper: hoàn tất, không lỗi runtime.")
