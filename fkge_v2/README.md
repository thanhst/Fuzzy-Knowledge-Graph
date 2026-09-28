# Bộ thực nghiệm FKG-E (KB1–KB6, Ablation, Baseline)

## Trạng thái so với thiết kế thực nghiệm ngày 26/09/2026

- Đã nối gói FRB BRSET fusion ảnh+bảng 5-fold tại
  `Source_code/data/exports/frb_patient_id_20260923`; manifest nguồn xác nhận
  `patient_overlap_count=0`.
- KB1 tách FISA tuần tự, FISA bảng tra, FKG-E không nhãn và FKG-E đầy đủ;
  báo AUC-ROC, AUC-PR, F1, Balanced Accuracy, độ trung thành, KL và thời gian.
- KB3-KB5 báo AUC trên validation folds, không dùng nhãn của outer test để
  chọn cấu hình. Đánh giá outer test còn chờ FRB dành riêng cho root test.
- KB6 benchmark trung vị 5 lần và báo hệ số góc log-log cho FISA tuần tự,
  FISA bảng tra và FKG-E.
- Mô hình hiện thực sự có `L_SGNS`, `L_node`, `L_inf`, `L_pred`, L2 và
  weighted pooling. `L_edge`, `L_A`, `L_B`, `L_rule` độc lập và attention
  pooling trong bản thiết kế mở rộng chưa có công thức/cài đặt tương ứng;
  output ghi rõ là chưa triển khai, không dùng tên thay thế.
- DeepWalk/TransE/DistMult tự viết mang hậu tố `-lite` và chỉ dùng smoke test.
  Node2Vec chuẩn, XGBoost và KGE chuẩn bằng PyKEEN chưa sẵn sàng trong môi
  trường hiện tại nên không được tính là baseline chính thức.
- Mỗi lần chạy qua `run_all.py` ghi vào `outputs/runs/<run_id>` cùng
  `run_manifest.json`, tránh trộn kết quả giữa các lần chạy.
- `fkge_v2` là bộ huấn luyện/đánh giá FKG-E trên đầu ra FRB của pipeline
  FKG-MM. Nó không thay thế phần xây dựng và đánh giá FKG-MM theo KB-M1--KB-M6.
  Đầu vào BRSET mặc định là modality `fusion`; token giữ nhãn `image::` hoặc
  `table::`, và cạnh được đánh dấu nội mô thức hoặc liên mô thức.

Bộ code này cung cấp khung thực nghiệm định lượng cho FKG-E (Mục 3.5.4
luận án), dùng FKG-MM trên BRSET (hoặc FKGS đã lấy mẫu ở Chương 2) làm
đầu vào và so sánh với FISA. Các thành phần loss và baseline còn thiếu
được liệt kê ở phần trạng thái phía trên và chưa được xem là kết quả chính thức.

## ⚠️ ĐỌC TRƯỚC KHI CHẠY — 5 phát hiện quan trọng trong quá trình xây dựng

Trong lúc viết và **kiểm thử thật** bộ code này (không chỉ viết cho chạy
được mà chủ động dò lỗi bằng dữ liệu tổng hợp có cấu trúc biết trước), đã
phát hiện và sửa 5 vấn đề — bạn cần biết để không hiểu nhầm khi chạy trên
dữ liệu thật của mình:

### 0. (Quan trọng nhất) Ban đầu KHÔNG có k-fold, KHÔNG có patient_id
Phiên bản đầu tiên chỉ chia dữ liệu **một lần duy nhất** theo thứ tự
(`samples[:60%], samples[60%:]`) — vi phạm trực tiếp Mục 3.5.1 Chương 3
(nguyên tắc chia dữ liệu theo bệnh nhân để tránh rò rỉ). **Đã bổ sung**:
- `data/kfold_utils.py`: Stratified Group K-Fold theo `patient_id`
  (dùng `sklearn.model_selection.StratifiedGroupKFold`), đúng Công thức
  (3.110): $P_{train}^{(k)} \cap P_{test}^{(k)} = \emptyset$ với mọi fold.
- **Tự động kiểm tra rò rỉ** sau mỗi lần chia (`_verify_no_patient_leakage`)
  — đã kiểm thử bằng cách CỐ TÌNH tạo dữ liệu rò rỉ để xác nhận cơ chế
  bắt lỗi hoạt động thật, không chỉ "chạy qua".
- **KB1/KB2** (so sánh chính, cần độ tin cậy cao nhất): chạy **đầy đủ cả
  5 fold**, tổng hợp mean±std qua (fold × seed).
- **KB3/KB4/KB5/Ablation/Baseline** (quét nhiều tổ hợp siêu tham số): dùng
  **1 fold cố định** (fold 0) để tiết kiệm thời gian tính toán — đúng tinh
  thần "inner loop" của nested cross-validation (Mục 3.5.2), vẫn đảm bảo
  patient-aware, không rò rỉ.
- Định dạng file test **bắt buộc** phải có trường `"patient_id"` cho mỗi
  mẫu — nếu thiếu, `make_patient_kfold()` sẽ dừng ngay với lỗi rõ ràng
  thay vì âm thầm coi mỗi mẫu là một bệnh nhân riêng.

**Trong lúc thêm patient_id vào bộ sinh dữ liệu tổng hợp, phát hiện thêm
một lỗi phụ**: công thức tạo "thiên hướng bệnh nhân" (patient bias) ban
đầu vô tình làm mức "High" thắng áp đảo 62.5% thay vì ~33% công bằng,
khiến FISA (dựa vào luật đã mine từ phân phối gốc) đột nhiên cho kết quả
rất tệ (0.27–0.53) trên dữ liệu test đã bị lệch phân phối. Đã sửa lại
công thức, kiểm chứng lại cho tỷ lệ thắng cân bằng đúng ~33% mỗi mức.

### 1. Mất cân bằng giữa các thành phần hàm mất mát (đã sửa)
Ban đầu, `L_SGNS` (có hàng trăm cặp token/batch) áp đảo hoàn toàn
`L_pred`/`L_inf` (chỉ vài mẫu/batch) về độ lớn tuyệt đối — dù cả hai có
trọng số λ/β/γ/δ = nhau, `L_SGNS` chiếm tới **99.5% tổng loss**. Hậu quả:
mô hình học representation đồng xuất hiện tốt nhưng **không học được tín
hiệu phân loại**, khiến $p_E$ suy biến về đúng tỉ lệ nhãn trung bình
(luôn đoán lớp đa số) bất kể siêu tham số. **Đã sửa** bằng cách chuẩn hóa
mỗi thành phần loss theo số phần tử của nó (trung bình, không phải tổng)
trong `models/fkge.py`. Sau khi sửa, cần **`delta_pred` lớn hơn đáng kể**
so với `beta_rule`/`lam_node` (giá trị mặc định trong `config.py` đã được
hiệu chỉnh dựa trên thực nghiệm: `delta_pred=20.0` so với `beta_rule=1.0`).

**Việc bạn cần làm trên dữ liệu thật:** mọi lần đánh giá đều tự động in
cảnh báo nếu accuracy không vượt rõ baseline "luôn đoán lớp đa số" (xem
`models/fisa.py::warn_if_not_beating_majority`). Nếu thấy cảnh báo này
trên BRSET thật, **tăng `delta_pred`/`gamma_inf`** trước tiên (thử
`[5, 10, 20, 50]`), sau đó mới tăng `lr`/`epochs`.

### 2. Seed chưa kiểm soát toàn bộ tính ngẫu nhiên (đã sửa)
`np.random.choice` (negative sampling) ban đầu dùng trạng thái ngẫu nhiên
**toàn cục**, không bị chi phối bởi tham số `seed` truyền vào từng model
— khiến việc lặp `n_seeds` để đo phương sai trở nên vô nghĩa (độ lệch
chuẩn luôn ra 0.0000 một cách đáng ngờ). **Đã sửa** bằng cách gọi
`random.seed()`/`np.random.seed()` ở đầu `fit()`.

### 3. FISA gần như phẳng theo |R|, FKG-E mới là đại lượng tăng (KB6)
Giả thuyết "ngây thơ" ban đầu (FISA tăng tuyến tính theo số luật, FKG-E
phẳng) **không đúng với cách cài đặt FISA có tiền tính ma trận trọng số
W một lần** (đúng Mệnh đề 3.2.11 trong Chương 3): sau khi `fit()`, chi phí
suy diễn FISA mỗi truy vấn gần như không đổi theo `|R|`; ngược lại, FKG-E
phải so khớp với toàn bộ `|R|` luật mỗi truy vấn nên **chi phí FKG-E mới
là đại lượng tăng theo `|R|`**. Kết quả KB6 cần được đọc theo đúng chiều
này — đây không phải lỗi.

---

## -3. SỬA LỖI HỆ THỐNG NGHIÊM TRỌNG: "Chế độ A" (FKG cố định) đã bị LOẠI BỎ HOÀN TOÀN

**Phát hiện qua kiểm tra chéo với `fkgs_v2`**: dù `data/pipeline_interface.py`
đã có cơ chế khai phá lại luật đúng theo fold ("Chế độ B") từ trước, **cả 5
script thực nghiệm** (`kb1_kb2`, `kb3_kb5`, `kb4_kb6`, `ablation`,
`baseline_comparison`) **vẫn gọi mặc định với `pipeline=None`** — tức trên
thực tế luôn chạy ở "Chế độ A": nạp **một FKG-MM cố định** từ file luật đã
khai phá sẵn, chỉ chia k-fold ở mức *chia mẫu test/train cho bước suy diễn*,
**không hề khai phá lại luật/mờ hoá theo từng fold**. Điều này khiến `FKG-E`
được xây trên `fkg` cố định, KHÔNG PHẢI trên FKG-MM khai phá riêng từ đúng
train của mỗi fold — vi phạm chính nguyên tắc Mục 3.5.1 mà README đã đề ra.

**Đã sửa triệt để**: xoá bỏ hoàn toàn tham số `fkg` cố định khỏi
`run_one_dataset()` và mọi hàm `_get_brset_or_synthetic()`. Giờ đây:
- Đầu vào là **dữ liệu THÔ** (`data/fkg_io.py::load_raw_records()` /
  `generate_synthetic_raw_records()`) — features số + `patient_id` + label,
  **chưa mờ hoá, chưa khai phá luật**.
- `pipeline` (kiểu `RuleMiningPipeline`) là **tham số bắt buộc** — nếu thiếu,
  `run_one_dataset()` **raise lỗi ngay**, không còn fallback âm thầm.
- Với MỖI fold: `pipeline.fit_and_mine(train_raw)` mờ hoá + khai phá luật
  **chỉ từ train của fold đó**, `pipeline.transform(test_raw)` mờ hoá test
  bằng đúng tham số đã fit. `FKGE(fkg_fold, ...)` giờ LUÔN nhận `fkg_fold`
  vừa khai phá — không bao giờ một FKG cố định.
- Thêm `RealisticSyntheticPipeline` (khác `SyntheticRuleMiningPipeline` cũ
  — vốn sinh luật *ngẫu nhiên theo seed*, không phản ánh nội dung dữ liệu):
  pipeline mới **thực sự** học ngưỡng mờ hoá (phân vị) từ train và đếm
  support/confidence thật để sinh luật.
- KB2 (so sánh FKG-MM đầy đủ vs FKGS đã nén) giờ mô phỏng nén luật bằng
  `RealisticSyntheticPipeline(sample_ratio=0.3)` — lấy mẫu con luật **ngay
  sau khi khai phá đầy đủ trên train của mỗi fold**, không rò rỉ.

**Bug phụ tìm thấy và sửa trong lúc viết `RealisticSyntheticPipeline`**:
nhãn ban đầu ở dạng số nguyên (`0`/`1`) trong khi FISA/FKG-E dự đoán chuỗi
(`"class-0"`/`"class-1"`) — khiến accuracy = 0.0000 giả tạo do so sánh sai
kiểu dữ liệu, không phải lỗi suy diễn. Đã sửa để nhãn luôn ở đúng định dạng
`"class-{label}"` xuyên suốt pipeline.

**Đã kiểm thử lại toàn bộ 5 script** sau khi sửa: số luật khai phá được
**khác nhau thật** giữa các fold (186, 183, 183, 186, 188 — không phải một
số cố định lặp lại), FISA/FKG-E cho kết quả có ý nghĩa (0.60–0.84), FKG-E
vượt rõ các baseline KGE chuẩn qua PyKEEN (TransE 0.62, DistMult 0.62,
ComplEx 0.56, RotatE 0.65 so với FKG-E 0.84).

### Nếu dữ liệu BRSET của bạn ĐÃ LÀ luật FRB sau khi fusion ảnh+bảng theo patient_id

Không cần viết pipeline mới — đã có sẵn **`PrefuzzifiedRulePipeline`**
(`data/pipeline_interface.py`), dùng khi mỗi bản ghi **đã mờ hoá sẵn**
(có `antecedent_tokens`, không phải số thô cần học ngưỡng phân vị như
`RealisticSyntheticPipeline` dùng cho Diabetes).

**Đặt file**: `data_real/brset_raw.json`, định dạng:
```json
{
  "records": [
    {"antecedent_tokens": ["Age-High", "HbA1c-High", "GLCM_Contrast-Medium"],
     "label": 1, "patient_id": "P00123"},
    {"antecedent_tokens": ["Age-Medium", "HbA1c-Low", "GLCM_Contrast-High"],
     "label": 0, "patient_id": "P00124"}
  ]
}
```
`kb1_kb2.py::run_kb1()`/`run_kb2()` đã tự động dùng đúng pipeline này cho
BRSET (nhận diện qua tên dataset bắt đầu bằng "BRSET").

**⚠️ Hạn chế đã biết (CÙNG LOẠI đã xác nhận "Hướng A" ở `fkgs_v2`)**:
`PrefuzzifiedRulePipeline` đếm support/confidence theo tần suất tuyệt đối,
KHÔNG chuẩn hoá theo $|R_l|$ từng lớp — trên dữ liệu mất cân bằng lớp mạnh
(đã kiểm chứng thực nghiệm với tỉ lệ 62%/38%), FISA có thể sụp về đúng
baseline lớp đa số. `warn_if_not_beating_majority()` tự động cảnh báo mỗi
khi việc này xảy ra khi bạn chạy trên BRSET thật — đọc kỹ log console.

### Nếu bạn tự viết pipeline riêng (ví dụ cần trích GLCM trực tiếp từ ảnh gốc)

Viết một lớp con kế thừa `RuleMiningPipeline` (xem `PrefuzzifiedRulePipeline`
hoặc `RealisticSyntheticPipeline` làm ví dụ) gọi đúng pipeline khai phá
luật GLCM/Wang-Mendel thật của bạn.

## -2. FISA đã sửa lại đúng công thức gốc (1.18)-(1.20) — thay đổi quan trọng

Bản trước của `models/fisa.py` dùng $C_{il}(x) = \sum_{v\in V_i}\mu_v(x_i)\,W_{vl}$
— nhân **độ thuộc mờ liên tục** $\mu_v(x_i)$ vào tổng, một số hạng
**không xuất hiện** trong công thức FISA gốc (1.18): $C_{il}=\sum_t B^t_{il}$
("tổng trên toàn bộ các **cung liên quan**").

**Đã sửa**: đọc đúng chữ "liên quan" — $C_{il}(x)$ phải là tổng **có điều
kiện lọc theo giá trị khớp** (so khớp rời rạc, không phải trọng số mờ
liên tục). Cài đặt bằng bảng tra cứu ba chiều `W[thuộc_tính][giá_trị][nhãn]`
(khớp đúng Chương 3, Công thức 3.67-3.68):

```python
W[attr][v][l] = Σ_{luật t có token tiền đề v} (support_t × confidence_t)   # fit(), XẤP XỈ
C[attr][l]    = W[attr][ v* ][l]   # v* = giá trị THẮNG CUỘC (độ thuộc lớn nhất), TRA CỨU trực tiếp
```

khác với trước đây (`C[attr][l] = Σ_v μ_v(x_i) × W[v][l]` — cộng dồn qua
mọi giá trị, nhân trọng số mờ). **Đã kiểm thử lại toàn bộ**: FISA mới cho
accuracy hợp lý trên dữ liệu tổng hợp (vượt rõ baseline lớp đa số ở hầu
hết cấu hình), tích hợp đúng với FKG-E (`L_inf` dùng `predict_proba_one()`
— interface không đổi) và với k-fold theo `patient_id` (đã chạy lại toàn
bộ `run_one_dataset()` qua `SyntheticRuleMiningPipeline`, xác nhận không
rò rỉ, luật khai phá lại đúng mỗi fold, FKG-E vẫn vượt baseline khi train
đủ epochs).

**Việc bạn cần làm nếu đã có ma trận $A$, $B$ thật** (Công thức 3.8-3.9,
tính từ pipeline khai phá luật Chương 2/3): thay bước `fit()` trong
`models/fisa.py` để dùng đúng $B^t_{il}$ thật thay vì xấp xỉ
`support × confidence` hiện tại — cấu trúc bảng tra cứu 3 chiều và bước
suy diễn (`_C`, `predict_one`) đã đúng, không cần sửa thêm.

## -1. K-fold nằm Ở ĐÂU trong quy trình nhiều bước (Hình 3.2 Chương 3)?

**Trả lời ngắn gọn: k-fold KHÔNG nằm ở một bước cụ thể nào trong hình —
nó nằm NGAY TRƯỚC TOÀN BỘ quy trình**, chia dữ liệu THÔ (chưa xử lý gì)
theo `patient_id`. Sau đó, **toàn bộ** các bước sau được chạy lại từ đầu
cho **mỗi fold**, chỉ dùng phần train của fold đó:

```
Ảnh BRSET + Dữ liệu lâm sàng (THÔ, có patient_id)
        │
        ▼
╔═══════════════════════════════════╗
║  K-FOLD SPLIT THEO patient_id      ║  <-- Ở ĐÂY, trước tất cả
║  (data/kfold_utils.py)             ║
╚═══════════════════════════════════╝
        │
        ▼ (lặp lại cho MỖI fold, chỉ dùng TRAIN)
  Chuẩn hoá, trích đặc trưng (GLCM/bảng)
        ▼
  Mờ hoá riêng từng mô thức (fit hàm thuộc CHỈ trên train)
        ▼
  Xây E_intra, E_cross
        ▼
  Sinh và SÀNG LỌC LUẬT đa mô thức   <-- luật khai phá LẠI mỗi fold,
        ▼                                KHÔNG dùng 1 luật cố định
  FKG-MM của fold này (chỉ từ train)
        │
        ├──────────────┐
        ▼              ▼
   FISA (test)     FKG-E (train để huấn luyện, test để đánh giá)
```

### ⚠️ Đã phát hiện: phiên bản đầu tiên của bộ code này làm SAI

Bản triển khai ban đầu chỉ lặp k-fold ở bước **cuối cùng** (FISA/FKG-E),
còn tái sử dụng **MỘT FKG-MM cố định** (nạp từ 1 file JSON) cho **mọi**
fold — nghĩa là bước mờ hoá và sinh luật chỉ chạy **một lần duy nhất**,
không nằm trong vòng lặp fold. Đây là **rò rỉ dữ liệu tinh vi**: nếu file
JSON đó được khai phá trên toàn bộ BRSET trước khi chia fold, tham số hàm
thuộc và chính tập luật đã "nhìn thấy" cả những bệnh nhân sau này rơi vào
tập test — dù không có nhãn nào bị sao chép trực tiếp.

**Đã sửa** bằng `data/pipeline_interface.py`: thêm giao diện
`RuleMiningPipeline` với hai phương thức `fit_and_mine(train_raw)` (khai
phá luật + fit tham số mờ hoá **chỉ từ train**) và `transform(test_raw)`
(mờ hoá test bằng đúng tham số đã fit, **không** fit lại). Hàm
`experiments/kb1_kb2.py::run_one_dataset()` giờ hỗ trợ **hai chế độ**:

- **Chế độ (B) — ĐÚNG, khuyến nghị**: truyền `pipeline=<RuleMiningPipeline
  của bạn>`. Với mỗi fold, luật được **khai phá lại từ đầu** chỉ từ train
  của fold đó. Đã kiểm thử: luật thực sự khác nhau giữa các fold (4/5 chữ
  ký luật khác biệt trong bài test), không rò rỉ patient_id.
- **Chế độ (A) — đơn giản hoá, có cảnh báo**: không truyền `pipeline`
  (mặc định), dùng 1 `fkg` cố định. **Chỉ hợp lệ nếu bạn tự đảm bảo** luật
  đó chưa từng "nhìn thấy" dữ liệu test — hệ thống tự in cảnh báo mỗi lần
  chế độ này được dùng.

### Bạn cần làm gì để dùng Chế độ (B) trên BRSET thật

Viết một lớp con kế thừa `RuleMiningPipeline` (xem `SyntheticRuleMiningPipeline`
trong `data/pipeline_interface.py` làm ví dụ), gọi **đúng pipeline khai
phá luật thật của bạn** (trích đặc trưng GLCM, mã hoá lâm sàng, xác định
tham số hàm thuộc tam giác, xây `E_intra`/`E_cross`, sinh luật Wang-Mendel
+ sàng lọc theo support/confidence) bên trong `fit_and_mine()`, chỉ dùng
đúng `train_raw_records` được truyền vào — không đụng đến dữ liệu ngoài
tham số đó.

```bash
python3 data/pipeline_interface.py   # kiểm thử: xác nhận luật khác nhau
                                       # giữa các fold + không rò rỉ
```

## 0. Chuyển đổi FKG sang KG chuẩn và chạy KG Embedding chuẩn (PyKEEN)

### 0.1. Vì sao cần chuyển đổi — hai khác biệt căn bản

| | FKG (đồ thị tri thức mờ) | KG chuẩn (đầu vào của TransE/DistMult/...) |
|---|---|---|
| Cạnh | Có trọng số mờ $\omega_{uv}\in[0,1]$ | Nhị phân: tồn tại hoặc không |
| Luật | n-ngôi: nhiều tiền đề → 1 hệ quả | Hai ngôi: đúng 1 head, 1 tail, 1 relation |

Không có phép chuyển đổi "đúng duy nhất" — luôn có đánh đổi giữa việc đơn
giản hoá và bảo toàn thông tin.

### 0.2. Ba chiến lược đã cài đặt (`data/fkg_to_kg.py`)

| Chiến lược | Cách làm | Ưu / nhược |
|---|---|---|
| `threshold` | $\omega_{uv}\ge$ ngưỡng → 1 cạnh `related_to` | Đơn giản nhất; mất hoàn toàn cấu trúc luật và độ tin cậy |
| `pairwise` | Mỗi luật tách thành cạnh `co_occurs` (giữa các tiền đề) + `implies` (tiền đề→hệ quả) | Dùng trong `models/kge_baselines.py`; hai luật khác nhau có thể vô tình tạo cùng 1 cặp |
| `reified` **(khuyến nghị)** | Mỗi luật là 1 thực thể ảo `rule_k`, nối tới từng tiền đề bằng `has_antecedent`, tới hệ quả bằng `has_consequent` | Bảo toàn đầy đủ cấu trúc n-ngôi của luật (kỹ thuật "reification" kinh điển trong RDF/KG) |

**Lưu ý về độ tin cậy mờ:** cả bốn mô hình TransE/DistMult/ComplEx/RotatE
nguyên bản đều **bỏ qua hoàn toàn** $\omega_{uv}$/confidence (chỉ học từ
facts nhị phân) — đây chính là nội dung Mệnh đề 3.2.17 (Chương 3) đã chỉ
ra: DistMult còn bị loại về mặt cấu trúc cho quan hệ bất đối xứng vì tính
đối xứng đại số $h^\top\text{diag}(r)t = t^\top\text{diag}(r)h$. Nếu cần
giữ lại thông tin mờ, hướng đúng về mặt lý thuyết là UKGE (Uncertain
Knowledge Graph Embedding, Chen et al. 2019) — nằm ngoài phạm vi 4
baseline chuẩn đã cài ở đây.

```bash
python3 data/fkg_to_kg.py   # demo nhanh, in số triples theo cả 3 chiến lược
```

### 0.3. Cài đặt PyKEEN — chạy TransE/DistMult/ComplEx/RotatE CHUẨN

Bản `models/kge_baselines.py` (Node2VecLite/TransELite/DistMultLite) là
code **tự viết, rút gọn** — đủ để kiểm tra pipeline nhưng **không nên
dùng làm baseline chính thức cho luận án** vì thiếu early stopping,
negative sampling chuẩn theo đúng bài báo gốc. Nên dùng **PyKEEN**
(Ali et al., "PyKEEN 1.0", *JMLR* 2021) — thư viện chuẩn, được cộng đồng
nghiên cứu KGE công nhận và trích dẫn rộng rãi:

```bash
pip install pykeen --break-system-packages   # kéo theo PyTorch, cần ~2-3GB trống
```

Sau khi cài, `experiments/baseline_comparison.py` **tự động phát hiện**
PyKEEN và chạy thêm TransE/DistMult/ComplEx/RotatE chuẩn (đánh dấu rõ
"(PyKEEN, chuẩn)" trong bảng kết quả) bên cạnh 5 phương pháp gốc — không
cần sửa gì thêm. Nếu chưa cài, script tự động bỏ qua phần này và chỉ
dùng bản "-lite", có in dòng nhắc rõ ràng.

Chạy thử riêng wrapper PyKEEN:
```bash
python3 models/kge_pykeen.py
```

**Bug thật đã phát hiện và sửa khi viết wrapper này:** ComplEx dùng
embedding **số phức** (chính phần ảo là lý do ComplEx biểu diễn được quan
hệ bất đối xứng — Mệnh đề 3.2.17). Nếu ép thẳng sang mảng NumPy thực,
Python **âm thầm cắt bỏ phần ảo** (đã quan sát `ComplexWarning` thật khi
chạy thử lần đầu), khiến ComplEx suy biến gần như DistMult. Đã sửa bằng
cách nối `[phần thực, phần ảo]` thành một vector thực có số chiều gấp đôi
trước khi đưa vào bộ phân loại kNN downstream.

## 1. Cài đặt

```bash
pip install numpy scikit-learn matplotlib --break-system-packages
```

Không cần PyTorch/TensorFlow — toàn bộ FKG-E được cài bằng NumPy thuần,
gradient viết tay và **đã kiểm chứng bằng gradient checking** (so khớp
đạo hàm giải tích với sai phân hữu hạn, xem `models/fkge.py::_gradient_check`).

## 2. Cấu trúc thư mục

```
fkge_v2/
├── config.py                    # TOÀN BỘ tham số — sửa đường dẫn dữ liệu ở đây
├── data/
│   └── fkg_io.py                 # Định dạng dữ liệu chuẩn + sinh dữ liệu tổng hợp
├── models/
│   ├── fisa.py                   # Suy luận FISA (Chương 2/3)
│   ├── fkge.py                   # FKG-E: SGNS + Node/Inf/Pred Loss (đã gradient-check)
│   └── kge_baselines.py          # Node2Vec-lite, TransE-lite, DistMult-lite + kNN
├── experiments/
│   ├── kb1_kb2.py                 # KB1: FKG-E vs FISA trên FKG gốc (3 bộ dữ liệu)
│   │                               # KB2: FKG-E vs FISA trên FKGS đã nén
│   ├── kb3_kb5.py                 # KB3: quét chiều nhúng d
│   │                               # KB5: quét (cửa sổ w, số mẫu âm K)
│   ├── kb4_kb6.py                 # KB4: quét lưới (λ, β) -> heatmap
│   │                               # KB6: đường cong khả năng mở rộng theo |R|
│   ├── ablation.py                 # Ablation: Rule-only / Node-only / Uniform / Full
│   └── baseline_comparison.py      # So sánh FISA/FKG-E/Node2Vec/TransE/DistMult
├── report/
│   └── generate_report.py          # Sinh bảng CSV/Markdown + biểu đồ PNG
├── run_all.py                       # Script tổng — chạy toàn bộ pipeline
└── outputs/                          # Kết quả (JSON, CSV, PNG, report_tong_hop.md)
```

## 3. Chuẩn bị dữ liệu đầu vào (BẮT BUỘC trước khi có kết quả thật)

Nếu bạn CHƯA chuẩn bị dữ liệu thật, **mọi script vẫn chạy được** — chúng
tự động dùng dữ liệu tổng hợp (synthetic) và **in cảnh báo rõ ràng** mỗi
lần dùng dữ liệu giả, để bạn không nhầm kết quả demo với kết quả thật.

### 3.1. Định dạng file luật (rules)

Tạo thư mục `data_real/` và đặt các file JSON theo định dạng sau (xem chi
tiết trong docstring của `data/fkg_io.py`):

```json
{
  "meta": {"dataset": "BRSET", "n_classes": 2, "class_names": ["No-DR", "DR"]},
  "vocab": ["Age-Low", "Age-Medium", "Age-High", "HbA1c-High", "class-DR", "class-No-DR"],
  "rules": [
    {
      "id": 0,
      "antecedent_tokens": ["Age-High", "HbA1c-High"],
      "consequent_token": "class-DR",
      "confidence": 0.87,
      "support": 0.12
    }
  ],
  "edges": [
    {"u": "Age-High", "v": "HbA1c-High", "mu": 0.63}
  ]
}
```

`rules` xuất từ chính pipeline khai phá luật FKG-MM/FKGS bạn đã xây ở
Chương 2–3 (đầu ra Wang–Mendel/M-CFIS). `edges` xuất từ ma trận quan hệ
mờ `A`/`ω_uv` (Mục 3.3.4). Nếu bạn chưa có bước xuất JSON, viết một script
nhỏ chuyển đổi từ cấu trúc luật nội bộ hiện có sang đúng định dạng trên —
đây là điểm tích hợp DUY NHẤT cần làm để dùng dữ liệu thật.

### 3.2. Định dạng file test (mẫu đã mờ hoá, có nhãn)

```json
{
  "samples": [
    {"membership": {"Age-High": 0.8, "HbA1c-High": 0.9}, "label": "class-DR"}
  ]
}
```

### 3.3. Đặt đúng đường dẫn trong `config.py`

Sửa 6 đường dẫn trong `class PATHS` — trỏ tới các file bạn vừa tạo:
`BRSET_RULES_FILE`, `DIABETES_KAGGLE_RULES_FILE`,
`HEALTHCARE_DIABETES_RULES_FILE`, `BRSET_FKGS_RULES_FILE` (luật đã lấy
mẫu bởi FKGS — Chương 2), và 3 file `*_TEST_FILE` tương ứng.

Với KB6 (cần nhiều mức nén 20/40/60/80/100%), nếu bạn có sẵn 5 file FKGS
thật ở các tỉ lệ khác nhau, đặt tên theo quy ước
`brset_fkgs_rules_20.json`, `..._40.json`, ... — script sẽ tự nhận diện.
Nếu không có, script tự lấy mẫu ngẫu nhiên từ FKG-MM đầy đủ để mô phỏng
(kém chính xác hơn dùng FKGS thật, nhưng đủ để có đường cong tham khảo).

## 4. Cách chạy

### Bước 1 — Kiểm tra pipeline chạy đúng (bắt buộc, ~5-10 phút)

```bash
python3 run_all.py --quick
python3 run_all.py --quick --brset-only  # chỉ dùng gói FRB BRSET thật
python3 run_all.py --quick --brset-only --epochs 1  # lượt kiểm chứng rất ngắn
```

Chạy với epochs/n_seeds giảm, dùng dữ liệu tổng hợp nếu chưa có dữ liệu
thật. Mục đích DUY NHẤT là xác nhận không có lỗi runtime — **không dùng
kết quả ở bước này để báo cáo**.

### Bước 2 — Chạy từng model riêng để kiểm tra (tuỳ chọn)

```bash
python3 models/fisa.py              # Kiểm tra FISA (in accuracy trên dữ liệu tổng hợp)
python3 models/fkge.py              # Gradient checking + huấn luyện thử FKG-E
python3 models/kge_baselines.py     # Kiểm tra Node2Vec/TransE/DistMult
```

### Bước 3 — Chạy đầy đủ trên dữ liệu thật

```bash
python3 run_all.py
```

Chạy toàn bộ KB1→KB6→Ablation→Baseline rồi tự động sinh báo cáo. Thời
gian phụ thuộc quy mô `|R|` thật của BRSET — với vài trăm luật, ước tính
30–90 phút cho toàn bộ (n_seeds=5, epochs=100 theo mặc định). Có thể chạy
từng phần:

```bash
python3 run_all.py --only kb1,kb2      # chỉ chạy KB1, KB2
python3 run_all.py --only kb6          # chỉ chạy KB6
```

### Bước 4 — Sinh lại báo cáo (nếu đã có sẵn outputs/*.json)

```bash
python3 report/generate_report.py
```

## 5. Đọc kết quả

Sau khi chạy xong, thư mục `outputs/` chứa:
- `*.json` — kết quả thô từng KB
- `table_*.csv` — bảng dạng CSV (mở bằng Excel)
- `figures/*.png` — biểu đồ so sánh
- **`report_tong_hop.md`** — báo cáo tổng hợp đầy đủ bảng + biểu đồ, dùng
  trực tiếp để trích vào Mục 3.5.4 của luận án

## 6. Các điểm cần LUÔN kiểm tra trước khi tin dùng một con số

1. **So với baseline lớp đa số**: mọi accuracy in ra kèm cảnh báo tự động
   nếu không vượt rõ baseline này. Đừng bỏ qua cảnh báo.
2. **So giữa FISA và FKG-E trên đúng cùng fold, cùng seed dữ liệu**: script
   đã đảm bảo điều này, nhưng nếu bạn sửa code, giữ nguyên tính chất này.
3. **Std giữa các seed phải khác 0** (trừ trường hợp đặc biệt): nếu thấy
   `std=0.0000` một cách hệ thống trên dữ liệu thật, nghi ngờ ngay có vấn
   đề tương tự phát hiện #2 ở trên.
4. **KB6**: đọc đúng chiều — FKG-E là đường tăng, FISA là đường phẳng, đây
   là hành vi ĐÚNG chứ không phải bug (xem phát hiện #3 ở trên).
