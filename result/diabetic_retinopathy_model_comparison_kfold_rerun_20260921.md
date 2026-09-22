# Diabetic Retinopathy Model Comparison - KFold Rerun

Nguồn deep baseline: `ROOT_DATA/train_test_selection/deep_baselines/kfold_rerun_20260921/summary.csv`.
Nguồn FKGS: `Source_code/data/result/KFold_feature_selection_rerun_20260921/kfold_fkgs_mean_std_summary.csv`.
Nguồn native FKG: `Source_code/data/result/KFold_feature_selection_rerun_20260921/kfold_modality_mean_std_summary.csv`.
Giao thức: patient-aware 5-fold validation, không dùng outer test trong lần tổng hợp này.

Lưu ý backbone ảnh: ResNet-50: `resnet50`; Early Fusion (MLP): `resnet50`; Late Fusion (Ensemble): `resnet50`.

| Mô hình | Kiểu dữ liệu | Acc (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1 (%) | AUC-ROC (%) | AUC-PR (%) | Train (s) | Test (s) | Total (s) | Ghi chú |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MLP | Tabular | 86.0 +/- 5.9 | 79.5 +/- 11.9 | 86.6 +/- 6.8 | 36.7 +/- 10.4 | 49.2 +/- 10.1 | 92.6 +/- 2.4 | 57.2 +/- 12.4 | 4.79 +/- 0.64 | 0.05 +/- 0.00 | 4.84 +/- 0.64 | runner kfold_rerun_20260921; device=cuda |
| ResNet-50 | Image | 69.6 +/- 11.4 | 55.8 +/- 18.4 | 70.8 +/- 13.7 | 16.6 +/- 8.0 | 23.7 +/- 6.2 | 73.0 +/- 4.4 | 25.7 +/- 10.4 | 277.98 +/- 265.26 | 2.41 +/- 2.08 | 280.39 +/- 267.34 | runner kfold_rerun_20260921; device=cuda; resnet_arch=resnet50 |
| Early Fusion (MLP) | Multimodal | 88.6 +/- 3.1 | 66.1 +/- 5.2 | 90.6 +/- 3.8 | 39.2 +/- 6.3 | 48.7 +/- 4.8 | 89.2 +/- 3.8 | 55.8 +/- 9.3 | 271.92 +/- 259.54 | 2.38 +/- 2.02 | 274.30 +/- 261.56 | runner kfold_rerun_20260921; device=cuda; resnet_arch=resnet50 |
| Late Fusion (Ensemble) | Multimodal | 86.8 +/- 5.8 | 74.1 +/- 18.7 | 87.9 +/- 7.6 | 40.0 +/- 15.8 | 49.1 +/- 9.8 | 90.9 +/- 2.4 | 49.7 +/- 15.9 | 261.47 +/- 244.53 | 1.38 +/- 0.04 | 262.85 +/- 244.54 | runner kfold_rerun_20260921; device=cuda; resnet_arch=resnet50 |
| FKG-UM (Bảng) | Unimodal FKG | 89.8 +/- 2.8 | 40.4 +/- 24.8 | 94.2 +/- 2.9 | 36.6 +/- 18.5 | 37.3 +/- 19.9 | 80.6 +/- 6.2 | 32.8 +/- 15.0 | 4.05 +/- 1.00 | 0.11 +/- 0.09 | 4.15 +/- 1.08 | native FKG rerun; folds=5; features=13 |
| FKG-UM (Ảnh) | Unimodal FKG | 74.3 +/- 5.5 | 33.0 +/- 20.6 | 77.9 +/- 7.4 | 10.8 +/- 4.3 | 16.0 +/- 7.3 | 58.6 +/- 8.7 | 12.4 +/- 3.7 | 3.68 +/- 0.22 | 0.01 +/- 0.00 | 3.69 +/- 0.22 | native FKG rerun; folds=5; features=7 |
| FKG-MM (đề xuất) | Multimodal FKG | 91.7 +/- 0.3 | 11.4 +/- 10.2 | 98.7 +/- 0.9 | 39.1 +/- 21.9 | 16.6 +/- 12.9 | 80.7 +/- 6.6 | 24.5 +/- 3.7 | 5.46 +/- 1.86 | 0.21 +/- 0.05 | 5.67 +/- 1.88 | native FKG rerun; folds=5; features=16 |
| FKG-MM (Filter) | Multimodal FKG | 91.7 +/- 0.6 | 13.5 +/- 9.7 | 98.6 +/- 0.9 | 44.5 +/- 18.6 | 19.7 +/- 11.9 | 76.0 +/- 4.7 | 20.6 +/- 4.5 | 5.29 +/- 0.31 | 0.17 +/- 0.01 | 5.46 +/- 0.32 | native FKG rerun; folds=5; features=16 |
| FKG-MM (Hadamard) | Multimodal FKG | 62.7 +/- 15.1 | 44.5 +/- 12.3 | 64.3 +/- 17.1 | 10.7 +/- 3.3 | 16.8 +/- 4.2 | 56.4 +/- 9.5 | 10.8 +/- 2.7 | 3.49 +/- 2.12 | 0.00 +/- 0.00 | 3.49 +/- 2.12 | native FKG rerun; folds=5; features=5 |
| FKG-MM (Tensor) | Multimodal FKG | 91.6 +/- 0.6 | 2.1 +/- 4.7 | 99.5 +/- 0.6 | 10.0 +/- 22.4 | 3.5 +/- 7.8 | 55.7 +/- 3.8 | 9.6 +/- 1.7 | 7.73 +/- 3.32 | 2.14 +/- 0.46 | 9.86 +/- 3.77 | native FKG rerun; folds=5; features=30 |
| FKG-MM (Wrapper) | Multimodal FKG | 83.7 +/- 8.1 | 31.7 +/- 13.7 | 88.2 +/- 9.2 | 23.1 +/- 12.0 | 24.7 +/- 11.4 | 63.2 +/- 12.4 | 18.5 +/- 11.0 | 29.25 +/- 3.76 | 0.01 +/- 0.00 | 29.26 +/- 3.76 | native FKG rerun; folds=5; features=6 |
| FKG-UM (Bảng) [FKG-S] | Unimodal FKG | 91.1 +/- 1.0 | 21.4 +/- 17.4 | 97.1 +/- 2.3 | 44.0 +/- 35.9 | 24.0 +/- 17.9 | 72.2 +/- 8.4 | 24.3 +/- 13.0 | 3.55 +/- 1.03 | 6.81 +/- 0.41 | 10.36 +/- 1.12 | config chosen by accuracy on these same 5 folds (optimistic); ran=15; epsilon=0.2; folds=5; features=13 |
| FKG-UM (Ảnh) [FKG-S] | Unimodal FKG | 70.6 +/- 10.8 | 36.8 +/- 25.9 | 73.5 +/- 13.7 | 9.8 +/- 4.7 | 15.1 +/- 7.9 | 55.6 +/- 9.6 | 11.6 +/- 3.8 | 2.52 +/- 0.11 | 4.70 +/- 0.15 | 7.22 +/- 0.25 | config chosen by accuracy on these same 5 folds (optimistic); ran=15; epsilon=0.2; folds=5; features=7 |
| FKG-MM (đề xuất) [FKG-S] | Multimodal FKG | 91.6 +/- 0.4 | 6.2 +/- 5.8 | 99.1 +/- 0.6 | 38.6 +/- 21.8 | 10.2 +/- 8.3 | 65.4 +/- 7.2 | 14.4 +/- 3.2 | 5.35 +/- 1.88 | 11.24 +/- 0.46 | 16.58 +/- 1.60 | config chosen by accuracy on these same 5 folds (optimistic); ran=15; epsilon=0.2; folds=5; features=16 |
| FKG-MM (Filter) [FKG-S] | Multimodal FKG | 91.9 +/- 0.5 | 4.2 +/- 2.3 | 99.5 +/- 0.3 | 46.7 +/- 36.1 | 7.5 +/- 4.2 | 65.1 +/- 3.1 | 13.1 +/- 1.6 | 6.52 +/- 0.17 | 15.82 +/- 1.05 | 22.33 +/- 1.05 | config chosen by accuracy on these same 5 folds (optimistic); ran=20; epsilon=0.2; folds=5; features=16 |
| FKG-MM (Hadamard) [FKG-S] | Multimodal FKG | 76.1 +/- 7.2 | 30.9 +/- 16.2 | 80.0 +/- 8.5 | 11.9 +/- 6.3 | 16.8 +/- 8.6 | 57.5 +/- 7.7 | 11.8 +/- 3.0 | 2.47 +/- 1.68 | 4.38 +/- 0.33 | 6.84 +/- 1.50 | config chosen by accuracy on these same 5 folds (optimistic); ran=15; epsilon=0.2; folds=5; features=5 |
| FKG-MM (Tensor) [FKG-S] | Multimodal FKG | 92.1 +/- 0.2 | 1.0 +/- 2.2 | 100.0 +/- 0.0 | 20.0 +/- 44.7 | 1.9 +/- 4.3 | 50.9 +/- 1.1 | 9.0 +/- 2.2 | 57.90 +/- 4.38 | 49.24 +/- 17.30 | 107.14 +/- 21.68 | config chosen by accuracy on these same 5 folds (optimistic); ran=15; epsilon=0.3; folds=5; features=30 |
| FKG-MM (Wrapper) [FKG-S] | Multimodal FKG | 81.1 +/- 6.4 | 42.0 +/- 16.5 | 84.5 +/- 7.1 | 22.3 +/- 13.4 | 27.5 +/- 11.3 | 64.7 +/- 12.7 | 21.8 +/- 10.0 | 28.08 +/- 3.74 | 3.87 +/- 0.20 | 31.95 +/- 3.78 | config chosen by accuracy on these same 5 folds (optimistic); ran=20; epsilon=0.3; folds=5; features=6 |

Sensitivity / Specificity / Precision / F1 là giá trị của lớp dương (diabetic retinopathy), không phải macro-average. Các cột macro_* nằm trong file CSV.

± là độ lệch chuẩn giữa 5 fold (ddof=1). AUC-PR của mô hình ngẫu nhiên bằng tỷ lệ lớp dương (~6.6%), không phải 50%.

## FKGS all ran/e tables

Bảng đầy đủ theo `ran` và `epsilon`: `Source_code/data/result/KFold_feature_selection_rerun_20260921/kfold_fkgs_tables.csv`.
