# Ghi chú source code cũ và các điểm cần cải tiến

File này ghi lại phần đánh giá dành cho bản triển khai cũ của thư mục
`fkge_v2`, trước các chỉnh sửa ổn định pipeline hiện tại.

## 1. Nguồn dữ liệu và vị trí k-fold

- Code cũ có dấu hiệu từng dùng một FKG/FKGS cố định cho nhiều fold. Cách
  này không phù hợp với thiết kế thí nghiệm vì tham số mờ hoá, support,
  confidence và tập luật có thể đã nhìn thấy dữ liệu test trước khi chia
  fold.
- Cần giữ nguyên nguyên tắc: chia theo `patient_id` trên dữ liệu đầu vào,
  sau đó khai phá luật lại bên trong từng train fold. Test fold chỉ được
  transform bằng tham số học từ train.
- Với BRSET đã được fusion/mờ hoá sẵn, đầu vào hợp lý là các bản ghi có
  `antecedent_tokens`, `label`, `patient_id`; không xử lý như dữ liệu số
  thô có `features`.

## 2. Điểm yếu của code cũ

- Một số script thí nghiệm trước đây chưa thống nhất loại pipeline: KB1/KB2
  dùng dữ liệu BRSET dạng luật FRB, nhưng KB3/KB4/KB5/KB6/ablation/baseline
  vẫn có đoạn giả định dữ liệu thô dạng số.
- KB6 còn phụ thuộc đường dẫn luật FKGS cố định đã bị bỏ khỏi `config.py`,
  nên có thể lỗi runtime khi chạy `run_all.py --only kb6`.
- Baseline KGE tự viết dạng `Lite` chỉ phù hợp để kiểm tra luồng chạy; nếu
  dùng cho báo cáo chính thức thì nên ưu tiên bản PyKEEN chuẩn hoặc ghi rõ
  đây là baseline rút gọn.
- Nhiều nhận xét dài nằm ngay trong docstring/comment của code, tốt cho
  quá trình debug nhưng làm file nguồn nặng và dễ khiến người đọc nhầm giữa
  thiết kế, cảnh báo và logic thực thi.
- Hai pipeline khai phá luật tạo `edges=[]`. Vì vậy `L_node` không nhận
  gradient; biến thể "bỏ L_node" và mô hình đầy đủ có thể thực chất là cùng
  một mô hình. Bảng ablation cũ vì thế không đủ giá trị chứng minh.
- KB3, KB4 và KB5 gọi tập ngoài cùng là `test` rồi dùng trực tiếp để so sánh
  cấu hình. Đây là chọn siêu tham số trên test, trái giao thức validation
  lồng trong tài liệu thiết kế.
- Bộ đo cũ chỉ có Accuracy/F1 macro/log-loss, thiếu AUC-ROC, AUC-PR,
  Balanced Accuracy, độ nhạy, độ đặc hiệu, độ đồng thuận với FISA và
  divergence KL. Do BRSET mất cân bằng mạnh, Accuracy có thể cao dù mô hình
  chỉ dự đoán lớp đa số.
- KB1 cũ chỉ có một FISA bảng tra và một FKG-E có nhãn; chưa tách FISA tuần
  tự, FISA bảng tra, FKG-E không nhãn và FKG-E đầy đủ như thiết kế mới.
- Ablation cũ chỉ có 4 dòng và gọi SGNS là `L_rule`, trong khi thiết kế mới
  tách `L_SGNS`, `L_edge`, `L_node`, `L_A`, `L_B`, `L_rule`, `L_inf`,
  `L_pred`, chuẩn hoá và pooling. Tên gọi cũ che mất các thành phần chưa hề
  được cài đặt.
- `Node2VecLite` dùng random walk không thiên lệch, gần với DeepWalk hơn
  Node2Vec vì không có hai tham số thiên lệch quay lại/khám phá. Không nên
  ghi dòng này là Node2Vec chuẩn trong bảng luận án.
- Tất cả JSON/CSV/biểu đồ ghi chung vào một thư mục. Chạy một phần có thể
  tạo báo cáo ghép kết quả mới với file cũ của KB khác, không còn truy vết
  được một lần chạy duy nhất.
- Lần chạy "full" trước vẫn dùng synthetic cho cả ba nguồn vì thư mục
  `data_real` trống, dù trong repository đã có gói FRB BRSET 5-fold theo
  bệnh nhân. Các con số đó chỉ là kiểm tra luồng, không phải kết quả thật.

## 3. Đánh giá chất lượng bản code cũ

- Điểm tốt: code có khung model/data/experiment/report, kiểm tra
  patient-aware split, seed và báo cáo Markdown/CSV/PNG.
- Điểm chưa tốt: tính nhất quán chưa cao giữa các script, còn sót tham
  chiếu cấu hình cũ, và có vài giả định dữ liệu ẩn. Đây là kiểu lỗi thường
  gặp khi code được mở rộng nhanh theo nhiều yêu cầu thí nghiệm liên tiếp.
- Nếu đánh giá mức "AI code tốt chưa": bản cũ là scaffold chạy được, nhưng
  chưa đạt mức mã thực nghiệm dùng cho luận án. Các lỗi `edges=[]`, chọn
  tham số trên test, thiếu AUC và gắn nhãn sai cho baseline có thể làm sai
  trực tiếp kết luận khoa học, không chỉ làm code khó bảo trì.

## 4. Việc nên làm tiếp

- Tách helper dùng chung cho việc nạp BRSET/synthetic và chọn pipeline, để
  tránh mỗi file experiment tự lặp một phiên bản `_get_brset_or_synthetic`.
- Thêm kiểm tra định dạng input: nếu record có `antecedent_tokens` thì dùng
  `PrefuzzifiedRulePipeline`; nếu có `features` thì dùng pipeline số thô;
  nếu thiếu `patient_id` thì dừng ngay.
- Khi có dữ liệu BRSET thật, chạy lại toàn bộ KB1-KB6 và ghi rõ trong report
  kết quả nào là synthetic, kết quả nào là dữ liệu thật.
- Nếu báo cáo chính thức cần baseline KGE, cài PyKEEN và dùng các dòng
  `TransE/DistMult/ComplEx/RotatE (PyKEEN, chuẩn)` thay cho bản `Lite`.
