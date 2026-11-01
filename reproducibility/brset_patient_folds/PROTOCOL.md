# Accurate descriptions of the two splitting methods

## Existing custom subset

The earlier export contained 1,529 available labeled images from 897 patients.
A separate patient-level hold-out contained 321 images from 168 patients. The
remaining root train contained 1,208 images from 729 patients and was partitioned
into five held-out folds by `assign_train_folds()`, with seed 42.

This function groups records by the true `patient_id`, orders groups using class
rarity, group size and a seeded tie-breaker, and allocates each group to the fold
that minimizes the imbalance objective for patient counts, image counts and
per-image label counts. It does not call scikit-learn's `StratifiedGroupKFold`.
Labels remain attached to individual images, including when a patient has
different labels across images.

For every fold, the patient sets of train and held-out are disjoint. Each of the
1,208 images appears in held-out exactly once across the five folds. These folds
were reproduced exactly from the repository source; their aggregate counts are:

| Fold | Train images | Held-out images | Held-out patients | Held-out positives |
|---|---:|---:|---:|---:|
| 1 | 966 | 242 | 146 | 20 |
| 2 | 966 | 242 | 146 | 19 |
| 3 | 966 | 242 | 146 | 19 |
| 4 | 967 | 241 | 146 | 20 |
| 5 | 967 | 241 | 145 | 19 |

Vietnamese wording for the existing partition:

> Bộ dữ liệu xuất trước đây giữ riêng một tập kiểm tra theo bệnh nhân. Trên phần
> huấn luyện còn lại gồm 1.208 ảnh của 729 bệnh nhân, năm fold được tạo bằng thuật
> toán phân bổ nhóm tự cài đặt, với seed 42. Mỗi nhóm gồm mọi ảnh của cùng một mã
> bệnh nhân; các nhóm được phân bổ để cân bằng gần đúng số bệnh nhân, số ảnh và số
> nhãn của từng lớp giữa các fold. Nhãn được giữ theo từng ảnh. Trong mỗi lượt,
> các bệnh nhân của fold tương ứng chỉ nằm trong tập giữ lại; mỗi ảnh xuất hiện
> đúng một lần trong các tập giữ lại sau năm lượt. Thuật toán này khác
> `StratifiedGroupKFold` của scikit-learn.

## SGKF over a defined input population

With `--method stratified-group`, the runner calls the repository's SGKF helper
on the retained input records: `groups = patient_id`, `y = per-image DR label`,
`n_splits = 5`, `shuffle = True`, `random_state = 42`. The input population must
be specified explicitly. The published verification checked all 16,266 records
from 8,524 patients in the complete local BRSET label table. It verified the
partition, not image availability or model performance.

Vietnamese wording for this alternative, after the corresponding experiment
actually uses these manifests:

> Sau bước lọc bản ghi, dữ liệu được chia thành năm fold bằng
> `StratifiedGroupKFold`, với khóa nhóm là `patient_id`, nhãn phân tầng là nhãn DR
> của từng ảnh, `shuffle=True` và `random_state=42`. Mọi ảnh của một bệnh nhân
> thuộc cùng một fold. Ở mỗi lượt, tập giữ lại gồm mọi ảnh thuộc fold đó và tập
> huấn luyện gồm bốn fold còn lại. Các tập bệnh nhân train và giữ lại không giao
> nhau; mỗi ảnh xuất hiện đúng một lần trong các tập giữ lại sau năm lượt.

Do not describe the old custom metrics as results from this new SGKF partition.

## Requirements for subsequent modeling

Within each outer fold, fit imputation, scaling, feature selection, learned
fusion, fuzzification, resampling, rule mining and graph construction using
outer train only. Apply fitted transformations to outer held-out records.

For validation-based hyperparameter selection, split inner train/validation by
true patient ID before fitting those steps; resample inner train only and keep
real-image validation separate from the learned graph and teacher targets.
Choose configuration and thresholds using that validation, then evaluate the
locked selection on outer held-out according to a declared refit procedure.
This publication verifies only ID splitting; it does not certify the previously
reviewed FKG-E training pipeline as satisfying these modeling requirements.
