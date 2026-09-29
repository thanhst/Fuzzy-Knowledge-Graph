# FKG-E BRSET fusion: đánh giá 5 fold × 5 seed, 100 epoch

Các lượt chạy đều dùng `quick=false`, cùng 5 fold theo patient ID và seed 42–46 trên validation của gói FRB BRSET fusion. Mỗi cấu hình FKG-E có 25 quan sát fold × seed, trừ khi ghi rõ khác. `*_std` là độ lệch chuẩn mẫu của quan sát; CI 95% lấy mẫu lại 10.000 lần theo fold sau khi trung bình seed trong fold. Đây chưa phải outer test hoặc mô hình Chương 3 đầy đủ.

| Nội dung | Kết quả chính | Hồ sơ |
|---|---|---|
| KB1 – FKG đầy đủ | FISA AUC 0,8231 ± 0,0791; FKG-E AUC 0,9305 ± 0,0170, BalAcc 0,8411 ± 0,0408; trung thành cân bằng 0,6685 | [KB1/KB2](tan_20260929_brset_fusion_kfold100_full/RUN_NOTES.md) |
| KB2 – mô phỏng 30% luật với hạn ngạch lớp | FKG-E AUC 0,9333 ± 0,0184; FISA rút gọn BalAcc 0,5000, nên độ trung thành chưa được xác nhận | [KB1/KB2](tan_20260929_brset_fusion_kfold100_full/RUN_NOTES.md) |
| Đường đánh đổi λ_I/λ_P | AUC 0,9300–0,9311; trung thành cân bằng 0,6633–0,6878 trên tỉ lệ 0–5; chưa đạt 0,9 | [Tradeoff](tan_20260929_brset_fusion_tradeoff_kfold100_retry/RUN_NOTES.md) |
| KB3 | Quét d=8,16,32,64,128; AUC 0,9305–0,9313; chọn d=8 theo quy tắc kích thước | [KB3/KB5](tan_20260929_brset_fusion_kfold100_scenarios/RUN_NOTES.md) |
| KB4 | Quét 30 cấu hình của năm trọng số đã cài đặt; λ_P chi phối AUC, λ_C có biến thiên AUC dưới 0,00003 trong dải thử | [KB4](tan_20260929_brset_fusion_kfold100_kb4/RUN_NOTES.md) |
| KB5 | Toàn luật: AUC 0,9305–0,9306; cửa sổ w=2: 0,9327–0,9328; K=2,5,10 gần như không đổi | [KB3/KB5](tan_20260929_brset_fusion_kfold100_scenarios/RUN_NOTES.md) |
| KB6 | Hệ số góc log–log: FISA tuần tự 0,9453; bảng tra 0,0017; FKG-E 0,0662 trên 5 mức số luật | [KB6/Baseline](tan_20260929_brset_fusion_kfold100_kb6_baseline_core_isolated/RUN_NOTES.md) |
| Ablation | Đầy đủ AUC 0,9305; chỉ `L_pred` 0,9303; bỏ `L_pred` 0,6206 | [Ablation](tan_20260929_brset_fusion_kfold100_ablation/RUN_NOTES.md) |
| Baseline | MLP AUC 0,9291 ± 0,0246, F1 0,5761; FKG-E AUC 0,9305 ± 0,0170, F1 0,5142. CI AUC chồng lấp | [KB6/Baseline](tan_20260929_brset_fusion_kfold100_kb6_baseline_core_isolated/RUN_NOTES.md) |

## Giới hạn cần nêu khi sử dụng

- Mã FKG-E hiện có `L_SGNS`, một phần `L_node`, `L_inf`, `L_pred` và L2, nhưng thiếu `L_edge`, `L_A`, `L_B`, `L_rule`. Vì vậy các bảng không chứng minh được đóng góp của bốn loss cấu trúc còn thiếu.
- Giáo viên FISA chưa hiệu chỉnh cả β/T theo inner validation tách bệnh nhân; ngưỡng quyết định cũng chưa được chọn theo inner validation. Độ trung thành chỉ khoảng 0,67–0,69 theo thước đo cân bằng.
- Các tập BRSET trong lượt này là validation của root train. Gói FRB chưa có outer test tương ứng. Không dùng các con số này làm kết luận cuối cho luận án.
- Các baseline hậu tố `-lite` không phải bản chuẩn; Node2Vec chuẩn chưa cài đặt, XGBoost và PyKEEN không có kết quả. MLP là mốc quan trọng: chênh lệch AUC so với FKG-E rất nhỏ và F1 cao hơn.
- Các run ID tách theo kịch bản để giữ đầu ra có manifest rõ ràng. Mỗi thư mục chứa JSON, CSV, hình và `RUN_NOTES.md`. Chi tiết CI, std và các quan sát fold/seed nằm trong JSON. KB6/baseline được đo trên logical CPU riêng lúc KB4 chạy trên lõi khác; diễn giải thời gian theo điều kiện đo này.
