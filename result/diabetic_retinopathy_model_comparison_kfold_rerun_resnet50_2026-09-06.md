# Diabetic Retinopathy Model Comparison - KFold Rerun - ResNet-50 Correction

Nguồn MLP/Early/Late deep baseline: `ROOT_DATA/train_test_selection/deep_baselines/kfold_rerun_cuda_20260905_2200/summary.csv`.
Nguồn ResNet-50: `ROOT_DATA/train_test_selection/deep_baselines/kfold_rerun_resnet50_cuda_20260906_2130/summary.csv`.
Nguồn FKGS: `Source_code/data/result/KFold_feature_selection_rerun_20260905/kfold_fkgs_mean_std_summary.csv` và `Source_code/data/result/KFold_feature_selection_rerun_20260905/kfold_fkgs_tables.csv`.
Giao thức: patient-aware 5-fold validation, không dùng outer test.

Lưu ý: hàng ResNet đã được chạy lại bằng `resnet_arch=resnet50`. Early/Late Fusion trong bảng này vẫn là kết quả full-run trước đó (`resnet_arch=resnet18`) vì chưa chạy lại toàn bộ fusion bằng ResNet-50.

| Mô hình | Kiểu dữ liệu | Protocol | Acc (%) | Precision (%) | Recall/Sensitivity (%) | Specificity (%) | F1 (%) | AUC (%) | Train (s) | Test (s) | Total (s) | Ghi chú |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MLP | Tabular | Patient-aware 5-fold validation; no outer test | 86.2 +/- 4.1 | 35.7 +/- 6.4 | 76.4 +/- 14.9 | 87.0 +/- 5.2 | 47.6 +/- 6.1 | 92.3 +/- 2.2 | 5.83 +/- 0.54 | 0.06 +/- 0.00 | 5.89 +/- 0.54 | runner kfold_rerun_cuda_20260905_2200; device=cuda |
| ResNet-50 | Image | Patient-aware 5-fold validation; no outer test | 78.8 +/- 8.4 | 19.3 +/- 4.9 | 43.8 +/- 22.5 | 81.8 +/- 10.8 | 24.2 +/- 4.8 | 69.8 +/- 4.8 | 393.74 +/- 296.14 | 2.37 +/- 2.02 | 396.11 +/- 297.52 | runner kfold_rerun_resnet50_cuda_20260906_2130; device=cuda; resnet_arch=resnet50 |
| Early Fusion (MLP) | Multimodal | Patient-aware 5-fold validation; no outer test | 86.8 +/- 6.1 | 40.4 +/- 12.5 | 77.4 +/- 7.3 | 87.7 +/- 7.2 | 51.1 +/- 9.6 | 92.3 +/- 1.0 | 219.38 +/- 69.18 | 1.80 +/- 0.46 | 221.17 +/- 69.63 | runner kfold_rerun_cuda_20260905_2200; device=cuda; resnet_arch=resnet18 |
| Late Fusion (Ensemble) | Multimodal | Patient-aware 5-fold validation; no outer test | 88.3 +/- 3.7 | 40.4 +/- 8.8 | 76.3 +/- 5.2 | 89.4 +/- 3.9 | 52.3 +/- 8.1 | 91.0 +/- 3.3 | 222.76 +/- 70.37 | 1.80 +/- 0.46 | 224.56 +/- 70.82 | runner kfold_rerun_cuda_20260905_2200; device=cuda; resnet_arch=resnet18 |
| FKG-UM (Ảnh) | Unimodal FKG | Patient-aware 5-fold validation; no outer test | 69.5 +/- 12.0 | 50.5 +/- 0.9 | 51.5 +/- 2.8 | ... | ... | ... | 32.56 +/- 2.35 | 28.40 +/- 2.29 | 60.96 +/- 3.61 | best accuracy from rerun; ran=15; epsilon=0.2; folds=5; features=7 |
| FKG-UM (Bảng) | Unimodal FKG | Patient-aware 5-fold validation; no outer test | 87.2 +/- 0.7 | 59.9 +/- 1.3 | 67.1 +/- 2.6 | ... | ... | ... | 340.57 +/- 2.35 | 92.54 +/- 5.69 | 433.11 +/- 7.71 | best accuracy from rerun; ran=20; epsilon=0.3; folds=5; features=13 |
| FKG-MM (đề xuất) | Multimodal FKG | Patient-aware 5-fold validation; no outer test | 90.5 +/- 0.9 | 65.0 +/- 1.2 | 69.7 +/- 1.3 | ... | ... | ... | 265.79 +/- 23.20 | 800.50 +/- 136.23 | 1066.28 +/- 151.31 | best accuracy from rerun; ran=15; epsilon=0.2; folds=5; features=16 |

## FKGS all ran/e tables

Bảng đầy đủ theo `ran` và `epsilon`: `Source_code/data/result/KFold_feature_selection_rerun_20260905/kfold_fkgs_tables.csv`.
