"""
data/fkg_to_kg.py — Chuyển đổi FKG (đồ thị tri thức MỜ, có độ thuộc liên
tục và luật n-ngôi IF-THEN) sang KG chuẩn (tập hợp bộ ba rời rạc (h, r, t))
mà các mô hình KG Embedding kinh điển (TransE, DistMult, ComplEx, RotatE)
yêu cầu làm đầu vào.

======================================================================
VÌ SAO CẦN CHUYỂN ĐỔI — HAI KHÁC BIỆT CĂN BẢN GIỮA FKG VÀ KG
======================================================================
1. CẠNH CÓ TRỌNG SỐ MỜ vs CẠNH NHỊ PHÂN
   FKG: mỗi cạnh (u,v) có độ tin cậy omega_uv thuộc [0,1] (Định nghĩa 3.2.2
   Chương 3). KG chuẩn: một bộ ba (h,r,t) CHỈ tồn tại hoặc KHÔNG tồn tại
   (nhị phân) — không có khái niệm "tồn tại 63%".

2. LUẬT N-NGÔI vs QUAN HỆ HAI NGÔI
   FKG: một luật r_k có DẠNG n-ngôi — nhiều tiền đề CÙNG LÚC suy ra một hệ
   quả: (A_1 AND A_2 AND ... AND A_n) => C. KG chuẩn: một bộ ba chỉ nối
   ĐÚNG HAI thực thể qua MỘT quan hệ — không biểu diễn trực tiếp được mối
   liên kết "nhiều-đối-một" của luật.

Vì vậy, KHÔNG có một phép chuyển đổi "đúng duy nhất" — có ĐÁNH ĐỔI giữa
các chiến lược. Module này cài đặt BA chiến lược, xếp theo mức độ bảo toàn
thông tin tăng dần:

  STRATEGY 1  "threshold"   : chỉ nhị phân hoá omega_uv >= ngưỡng -> cạnh
                                tồn tại. ĐƠN GIẢN NHẤT nhưng làm MẤT hoàn
                                toàn thông tin mức độ tin cậy và cấu trúc
                                luật (chỉ giữ quan hệ đôi một).
  STRATEGY 2  "pairwise"     : (đã dùng trong models/kge_baselines.py) mỗi
                                luật n-ngôi bị PHÂN RÃ thành các cặp đôi:
                                quan hệ "co_occurs" giữa các tiền đề, quan
                                hệ "implies" từ tiền đề tới hệ quả. Mất
                                thông tin "các tiền đề này cùng thuộc MỘT
                                luật cụ thể" (hai luật khác nhau có thể vô
                                tình tạo ra cùng một cặp "co_occurs").
  STRATEGY 3  "reified"      : mỗi LUẬT được coi là MỘT THỰC THỂ ẢO riêng
                                (rule_k), nối tới từng tiền đề bằng quan hệ
                                "has_antecedent" và tới hệ quả bằng quan hệ
                                "has_consequent". BẢO TOÀN ĐẦY ĐỦ cấu trúc
                                n-ngôi của luật (đây là kỹ thuật "reification"
                                kinh điển trong RDF/KG khi cần biểu diễn
                                quan hệ nhiều ngôi bằng các bộ ba hai ngôi).

KHUYẾN NGHỊ: dùng "reified" làm mặc định cho báo cáo chính thức (bảo toàn
thông tin tốt nhất, đúng bản chất luật FKG), dùng "pairwise" nếu muốn kết
quả có thể so sánh trực tiếp với các benchmark KGE cổ điển hoạt động trên
quan hệ đôi một đơn thuần.

VỀ ĐỘ TIN CẬY omega_uv/support/confidence: bị BỎ QUA hoàn toàn trong TransE/
DistMult/ComplEx/RotatE nguyên bản (chúng chỉ học từ facts nhị phân). Nếu
muốn giữ lại thông tin mờ này, xem ghi chú cuối file về UKGE (Uncertain
Knowledge Graph Embedding, Chen et al. 2019) — hướng mở rộng phù hợp nhất
về mặt lý thuyết cho FKG nhưng NẰM NGOÀI phạm vi bốn baseline chuẩn.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fkg_io import FKGRuleBase


def fkg_to_triples(fkg: FKGRuleBase, strategy="reified", edge_threshold=0.5):
    """Chuyển FKGRuleBase thành list các bộ ba (head, relation, tail) dạng
    chuỗi (str, str, str). `strategy` in {"threshold", "pairwise", "reified"}.
    """
    if strategy == "threshold":
        return _strategy_threshold(fkg, edge_threshold)
    elif strategy == "pairwise":
        return _strategy_pairwise(fkg)
    elif strategy == "reified":
        return _strategy_reified(fkg)
    else:
        raise ValueError(f"Không hỗ trợ strategy='{strategy}'. "
                          f"Chọn 'threshold', 'pairwise' hoặc 'reified'.")


def _strategy_threshold(fkg, threshold):
    triples = []
    for e in fkg.edges:
        if e["mu"] >= threshold:
            triples.append((e["u"], "related_to", e["v"]))
    return triples


def _strategy_pairwise(fkg):
    """Giống hệt _build_triples() trong models/kge_baselines.py — tách
    riêng ra đây để dùng chung cho cả PyKEEN lẫn bản -lite tự viết."""
    triples = []
    for r in fkg.rules:
        ante = r["antecedent_tokens"]
        cons = r["consequent_token"]
        for i in range(len(ante)):
            for j in range(len(ante)):
                if i != j:
                    triples.append((ante[i], "co_occurs", ante[j]))
            triples.append((ante[i], "implies", cons))
    return triples


def _strategy_reified(fkg):
    """Mỗi luật r_k trở thành một thực thể ảo 'rule_{k}', giữ nguyên đầy đủ
    liên kết n-ngôi tới TỪNG tiền đề và hệ quả của đúng luật đó."""
    triples = []
    for r in fkg.rules:
        rule_entity = f"rule_{r['id']}"
        for tok in r["antecedent_tokens"]:
            triples.append((rule_entity, "has_antecedent", tok))
        triples.append((rule_entity, "has_consequent", r["consequent_token"]))
        # Giữ thêm 1 quan hệ meta để KGE có thể học "mức độ mạnh" của luật
        # một cách RỜI RẠC (KHÔNG phải omega_uv liên tục, mà là 3 mức):
        # KGE nhị phân không nhận trọng số thực; đây là cách xấp xỉ thô để
        # không bỏ hoàn toàn thông tin confidence.
        conf_bucket = "conf_high" if r["confidence"] >= 0.75 else (
            "conf_medium" if r["confidence"] >= 0.5 else "conf_low")
        triples.append((rule_entity, "has_confidence", conf_bucket))
    return triples


def save_triples_tsv(triples, path):
    """Lưu triples ra file TSV (head\\trelation\\ttail), định dạng chuẩn mà
    PyKEEN's TriplesFactory.from_path() đọc trực tiếp được."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


if __name__ == "__main__":
    from data.fkg_io import generate_synthetic_fkg

    fkg = generate_synthetic_fkg(n_rules=50, seed=1)
    for strat in ["threshold", "pairwise", "reified"]:
        triples = fkg_to_triples(fkg, strategy=strat)
        n_entities = len({h for h, _, _ in triples} | {t for _, _, t in triples})
        n_relations = len({r for _, r, _ in triples})
        print(f"Chiến lược '{strat:10s}': {len(triples):5d} triples, "
              f"{n_entities:4d} thực thể, {n_relations} loại quan hệ")

    triples = fkg_to_triples(fkg, strategy="reified")
    save_triples_tsv(triples, "/tmp/test_triples.tsv")
    with open("/tmp/test_triples.tsv") as f:
        lines = f.readlines()
    assert len(lines) == len(triples), "Lỗi lưu/đọc file triples!"
    print(f"\nĐã lưu và đọc lại {len(lines)} dòng từ /tmp/test_triples.tsv -- OK")
