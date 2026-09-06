# Diabetic Retinopathy Model Comparison - KFold Rerun

Nguồn deep baseline: `ROOT_DATA/train_test_selection/deep_baselines/kfold_rerun_resnet50_cuda_20260906_2130/summary.csv`, `ROOT_DATA/train_test_selection/deep_baselines/kfold_rerun_cuda_20260905_2200/summary.csv`.
Nguồn FKGS: `Source_code/data/result/KFold_feature_selection_rerun_20260905/kfold_fkgs_mean_std_summary.csv`.
Nguồn native FKG: `Source_code/data/result/KFold_feature_selection_rerun_20260905/kfold_modality_mean_std_summary.csv`.
Giao thức: patient-aware 5-fold validation, không dùng outer test trong lần tổng hợp này.

Lưu ý backbone ảnh: ResNet-50: `resnet50`; Early Fusion (MLP): `resnet18`; Late Fusion (Ensemble): `resnet18`.

| Mô hình | Kiểu dữ liệu | Protocol | Acc (%) | Precision (%) | Recall/Sensitivity (%) | Specificity (%) | F1 (%) | AUC (%) | Train (s) | Test (s) | Total (s) | Ghi chú |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MLP | Tabular | Patient-aware 5-fold validation; no outer test | 86.2 +/- 4.1 | 35.7 +/- 6.4 | 76.4 +/- 14.9 | 87.0 +/- 5.2 | 47.6 +/- 6.1 | 92.3 +/- 2.2 | 5.83 +/- 0.54 | 0.06 +/- 0.00 | 5.89 +/- 0.54 | runner kfold_rerun_cuda_20260905_2200; device=cuda |
| ResNet-50 | Image | Patient-aware 5-fold validation; no outer test | 78.8 +/- 8.4 | 19.3 +/- 4.9 | 43.8 +/- 22.5 | 81.8 +/- 10.8 | 24.2 +/- 4.8 | 69.8 +/- 4.8 | 393.74 +/- 296.14 | 2.37 +/- 2.02 | 396.11 +/- 297.52 | runner kfold_rerun_resnet50_cuda_20260906_2130; device=cuda; resnet_arch=resnet50 |
| Early Fusion (MLP) | Multimodal | Patient-aware 5-fold validation; no outer test | 86.8 +/- 6.1 | 40.4 +/- 12.5 | 77.4 +/- 7.3 | 87.7 +/- 7.2 | 51.1 +/- 9.6 | 92.3 +/- 1.0 | 219.38 +/- 69.18 | 1.80 +/- 0.46 | 221.17 +/- 69.63 | runner kfold_rerun_cuda_20260905_2200; device=cuda; resnet_arch=resnet18 |
| Late Fusion (Ensemble) | Multimodal | Patient-aware 5-fold validation; no outer test | 88.3 +/- 3.7 | 40.4 +/- 8.8 | 76.3 +/- 5.2 | 89.4 +/- 3.9 | 52.3 +/- 8.1 | 91.0 +/- 3.3 | 222.76 +/- 70.37 | 1.80 +/- 0.46 | 224.56 +/- 70.82 | runner kfold_rerun_cuda_20260905_2200; device=cuda; resnet_arch=resnet18 |
| FKG-UM (Ảnh) | Unimodal FKG | Patient-aware 5-fold validation; no outer test | 67.0 +/- 10.7 | 50.4 +/- 0.4 | 51.6 +/- 1.7 | 69.3 +/- 12.4 | 45.4 +/- 3.2 | 49.3 +/- 2.8 | 15.46 +/- 4.10 | 0.21 +/- 0.24 | 15.68 +/- 4.34 | native FKG rerun; folds=5; features=7 |
| FKG-UM (Bảng) | Unimodal FKG | Patient-aware 5-fold validation; no outer test | 82.1 +/- 1.8 | 57.0 +/- 0.6 | 66.5 +/- 1.2 | 84.5 +/- 1.8 | 58.0 +/- 0.9 | 81.3 +/- 0.6 | 11.15 +/- 0.12 | 0.53 +/- 0.05 | 11.68 +/- 0.14 | native FKG rerun; folds=5; features=13 |
| FKG-MM (đề xuất) | Multimodal FKG | Patient-aware 5-fold validation; no outer test | 90.0 +/- 1.0 | 65.7 +/- 0.8 | 75.0 +/- 1.4 | 92.3 +/- 1.1 | 68.8 +/- 0.7 | 86.6 +/- 0.7 | 24.25 +/- 3.27 | 1.47 +/- 0.13 | 25.72 +/- 3.30 | native FKG rerun; folds=5; features=16 |

## FKGS all ran/e tables

Bảng đầy đủ theo `ran` và `epsilon`: `Source_code/data/result/KFold_feature_selection_rerun_20260905/kfold_fkgs_tables.csv`.
