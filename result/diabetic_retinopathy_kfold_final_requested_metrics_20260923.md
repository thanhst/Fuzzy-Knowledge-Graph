# KFold final requested metrics - 2026-09-23

AUC = AUC-ROC. Moi o do do co dang mean +/- std theo 5-fold.

## Bang tong hop cuoi

| Nhóm | Mô hình | Dữ liệu/chiến lược | ran | epsilon | Acc (%) | F1 (%) | AUC-ROC (%) | Spec (%) | Sens (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | MLP | Tabular |  |  | 86.0 ± 5.9 | 49.2 ± 10.1 | 92.6 ± 2.4 | 86.6 ± 6.8 | 79.5 ± 11.9 |
| Baseline | ResNet-50 | Image |  |  | 69.6 ± 11.4 | 23.7 ± 6.2 | 73.0 ± 4.4 | 70.8 ± 13.7 | 55.8 ± 18.4 |
| Fusion baseline | Early Fusion (MLP) | Multimodal |  |  | 88.6 ± 3.1 | 48.7 ± 4.8 | 89.2 ± 3.8 | 90.6 ± 3.8 | 66.1 ± 5.2 |
| Fusion baseline | Late Fusion (Ensemble) | Multimodal |  |  | 86.8 ± 5.8 | 49.1 ± 9.8 | 90.9 ± 2.4 | 87.9 ± 7.6 | 74.1 ± 18.7 |
| FKG-UM native | FKG-UM (Bảng) | Unimodal FKG |  |  | 89.8 ± 2.8 | 37.3 ± 19.9 | 80.6 ± 6.2 | 94.2 ± 2.9 | 40.4 ± 24.8 |
| FKG-UM native | FKG-UM (Ảnh) | Unimodal FKG |  |  | 74.3 ± 5.5 | 16.0 ± 7.3 | 58.6 ± 8.7 | 77.9 ± 7.4 | 33.0 ± 20.6 |
| FKG-MM native / 5 fusion strategies | FKG-MM (đề xuất) | Multimodal FKG |  |  | 91.7 ± 0.3 | 16.6 ± 12.9 | 80.7 ± 6.6 | 98.7 ± 0.9 | 11.4 ± 10.2 |
| FKG-MM native / 5 fusion strategies | FKG-MM (Filter) | Multimodal FKG |  |  | 91.7 ± 0.6 | 19.7 ± 11.9 | 76.0 ± 4.7 | 98.6 ± 0.9 | 13.5 ± 9.7 |
| FKG-MM native / 5 fusion strategies | FKG-MM (Hadamard) | Multimodal FKG |  |  | 62.7 ± 15.1 | 16.8 ± 4.2 | 56.4 ± 9.5 | 64.3 ± 17.1 | 44.5 ± 12.3 |
| FKG-MM native / 5 fusion strategies | FKG-MM (Tensor) | Multimodal FKG |  |  | 91.6 ± 0.6 | 3.5 ± 7.8 | 55.7 ± 3.8 | 99.5 ± 0.6 | 2.1 ± 4.7 |
| FKG-MM native / 5 fusion strategies | FKG-MM (Wrapper) | Multimodal FKG |  |  | 83.7 ± 8.1 | 24.7 ± 11.4 | 63.2 ± 12.4 | 88.2 ± 9.2 | 31.7 ± 13.7 |
| FKG-UM/FKG-MM 15%-20% chọn theo KFold | FKG-UM (Bảng) | Unimodal FKG | 15 | 0.2 | 91.1 ± 1.0 | 24.0 ± 17.9 | 72.2 ± 8.4 | 97.1 ± 2.3 | 21.4 ± 17.4 |
| FKG-UM/FKG-MM 15%-20% chọn theo KFold | FKG-UM (Ảnh) | Unimodal FKG | 15 | 0.2 | 70.6 ± 10.8 | 15.1 ± 7.9 | 55.6 ± 9.6 | 73.5 ± 13.7 | 36.8 ± 25.9 |
| FKG-UM/FKG-MM 15%-20% chọn theo KFold | FKG-MM (đề xuất) | Multimodal FKG | 15 | 0.2 | 91.6 ± 0.4 | 10.2 ± 8.3 | 65.4 ± 7.2 | 99.1 ± 0.6 | 6.2 ± 5.8 |
| FKG-UM/FKG-MM 15%-20% chọn theo KFold | FKG-MM (Filter) | Multimodal FKG | 20 | 0.2 | 91.9 ± 0.5 | 7.5 ± 4.2 | 65.1 ± 3.1 | 99.5 ± 0.3 | 4.2 ± 2.3 |
| FKG-UM/FKG-MM 15%-20% chọn theo KFold | FKG-MM (Hadamard) | Multimodal FKG | 15 | 0.2 | 76.1 ± 7.2 | 16.8 ± 8.6 | 57.5 ± 7.7 | 80.0 ± 8.5 | 30.9 ± 16.2 |
| FKG-UM/FKG-MM 15%-20% chọn theo KFold | FKG-MM (Tensor) | Multimodal FKG | 15 | 0.3 | 92.1 ± 0.2 | 1.9 ± 4.3 | 50.9 ± 1.1 | 100.0 ± 0.0 | 1.0 ± 2.2 |
| FKG-UM/FKG-MM 15%-20% chọn theo KFold | FKG-MM (Wrapper) | Multimodal FKG | 20 | 0.3 | 81.1 ± 6.4 | 27.5 ± 11.3 | 64.7 ± 12.7 | 84.5 ± 7.1 | 42.0 ± 16.5 |

## Chi tiet FKG-UM/FKG-MM theo ran va epsilon

| Bảng | Mô hình | Dữ liệu/chiến lược | ran | epsilon | Acc (%) | F1 (%) | AUC-ROC (%) | Spec (%) | Sens (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bang 3.2 | FKG-UM | Du lieu dang bang full | 15 | 0.2 | 91.1 ± 1.0 | 24.0 ± 17.9 | 72.2 ± 8.4 | 97.1 ± 2.3 | 21.4 ± 17.4 |
| Bang 3.2 | FKG-UM | Du lieu anh | 15 | 0.2 | 70.6 ± 10.8 | 15.1 ± 7.9 | 55.6 ± 9.6 | 73.5 ± 13.7 | 36.8 ± 25.9 |
| Bang 3.2 | FKG-MM | Du lieu anh+bang | 15 | 0.2 | 91.6 ± 0.4 | 10.2 ± 8.3 | 65.4 ± 7.2 | 99.1 ± 0.6 | 6.2 ± 5.8 |
| Bang 3.2 | FKG-MM | Fusion Filter | 15 | 0.2 | 91.6 ± 0.6 | 5.5 ± 5.0 | 61.7 ± 4.5 | 99.3 ± 0.7 | 3.1 ± 2.8 |
| Bang 3.2 | FKG-MM | Fusion Hadamard | 15 | 0.2 | 76.1 ± 7.2 | 16.8 ± 8.6 | 57.5 ± 7.7 | 80.0 ± 8.5 | 30.9 ± 16.2 |
| Bang 3.2 | FKG-MM | Fusion Tensor | 15 | 0.2 | 91.6 ± 0.6 | 0.0 ± 0.0 | 50.2 ± 0.4 | 99.6 ± 0.6 | 0.0 ± 0.0 |
| Bang 3.2 | FKG-MM | Fusion Wrapper | 15 | 0.2 | 80.1 ± 7.4 | 25.2 ± 12.7 | 65.9 ± 12.5 | 83.8 ± 8.0 | 38.0 ± 15.0 |
| Bang 3.3 | FKG-UM | Du lieu dang bang full | 15 | 0.3 | 90.5 ± 1.8 | 23.8 ± 18.2 | 73.6 ± 7.7 | 96.3 ± 3.4 | 23.8 ± 24.5 |
| Bang 3.3 | FKG-UM | Du lieu anh | 15 | 0.3 | 65.2 ± 3.9 | 14.9 ± 6.3 | 56.7 ± 11.4 | 67.4 ± 5.0 | 39.1 ± 19.3 |
| Bang 3.3 | FKG-MM | Du lieu anh+bang | 15 | 0.3 | 91.6 ± 0.5 | 3.6 ± 4.9 | 60.7 ± 6.8 | 99.4 ± 0.7 | 2.1 ± 2.8 |
| Bang 3.3 | FKG-MM | Fusion Filter | 15 | 0.3 | 91.6 ± 0.3 | 8.6 ± 8.5 | 58.3 ± 2.9 | 99.2 ± 0.5 | 5.2 ± 5.1 |
| Bang 3.3 | FKG-MM | Fusion Hadamard | 15 | 0.3 | 74.1 ± 7.8 | 16.4 ± 4.6 | 57.8 ± 5.3 | 77.7 ± 9.4 | 32.9 ± 13.8 |
| Bang 3.3 | FKG-MM | Fusion Tensor | 15 | 0.3 | 92.1 ± 0.2 | 1.9 ± 4.3 | 50.9 ± 1.1 | 100.0 ± 0.0 | 1.0 ± 2.2 |
| Bang 3.3 | FKG-MM | Fusion Wrapper | 15 | 0.3 | 78.4 ± 7.1 | 18.8 ± 9.2 | 59.3 ± 11.1 | 82.7 ± 7.7 | 28.8 ± 10.6 |
| Bang 3.4 | FKG-UM | Du lieu dang bang full | 20 | 0.2 | 90.1 ± 2.4 | 26.9 ± 22.6 | 75.1 ± 9.4 | 95.6 ± 2.2 | 27.0 ± 24.6 |
| Bang 3.4 | FKG-UM | Du lieu anh | 20 | 0.2 | 66.2 ± 5.0 | 15.1 ± 6.8 | 53.9 ± 14.3 | 68.7 ± 5.8 | 37.9 ± 19.2 |
| Bang 3.4 | FKG-MM | Du lieu anh+bang | 20 | 0.2 | 91.6 ± 0.6 | 9.8 ± 12.6 | 69.4 ± 5.7 | 99.1 ± 0.4 | 6.2 ± 8.7 |
| Bang 3.4 | FKG-MM | Fusion Filter | 20 | 0.2 | 91.9 ± 0.5 | 7.5 ± 4.2 | 65.1 ± 3.1 | 99.5 ± 0.3 | 4.2 ± 2.3 |
| Bang 3.4 | FKG-MM | Fusion Hadamard | 20 | 0.2 | 68.6 ± 14.2 | 17.6 ± 4.1 | 58.2 ± 5.9 | 71.2 ± 16.0 | 39.0 ± 11.1 |
| Bang 3.4 | FKG-MM | Fusion Tensor | 20 | 0.2 | 91.7 ± 0.3 | 0.0 ± 0.0 | 50.4 ± 0.4 | 99.7 ± 0.2 | 0.0 ± 0.0 |
| Bang 3.4 | FKG-MM | Fusion Wrapper | 20 | 0.2 | 80.8 ± 5.8 | 25.6 ± 10.9 | 63.9 ± 12.2 | 84.3 ± 6.8 | 40.9 ± 21.0 |
| Bang 3.5 | FKG-UM | Du lieu dang bang full | 20 | 0.3 | 90.7 ± 1.6 | 26.6 ± 14.7 | 71.4 ± 8.3 | 96.5 ± 2.8 | 24.7 ± 19.3 |
| Bang 3.5 | FKG-UM | Du lieu anh | 20 | 0.3 | 70.1 ± 6.0 | 15.2 ± 7.4 | 56.2 ± 11.1 | 73.1 ± 7.7 | 35.9 ± 20.0 |
| Bang 3.5 | FKG-MM | Du lieu anh+bang | 20 | 0.3 | 91.4 ± 0.4 | 9.9 ± 8.3 | 66.3 ± 3.8 | 98.8 ± 0.5 | 6.2 ± 5.8 |
| Bang 3.5 | FKG-MM | Fusion Filter | 20 | 0.3 | 91.7 ± 0.5 | 10.2 ± 8.3 | 62.6 ± 2.1 | 99.2 ± 0.7 | 6.2 ± 5.8 |
| Bang 3.5 | FKG-MM | Fusion Hadamard | 20 | 0.3 | 67.7 ± 14.2 | 15.5 ± 6.1 | 51.9 ± 10.2 | 70.7 ± 15.5 | 34.0 ± 12.0 |
| Bang 3.5 | FKG-MM | Fusion Tensor | 20 | 0.3 | 91.7 ± 0.3 | 0.0 ± 0.0 | 50.4 ± 0.3 | 99.7 ± 0.2 | 0.0 ± 0.0 |
| Bang 3.5 | FKG-MM | Fusion Wrapper | 20 | 0.3 | 81.1 ± 6.4 | 27.5 ± 11.3 | 64.7 ± 12.7 | 84.5 ± 7.1 | 42.0 ± 16.5 |

## Nguon du lieu

- `result/diabetic_retinopathy_model_comparison_kfold_rerun_20260921.csv`
- `Source_code/data/result/KFold_feature_selection_rerun_20260921/kfold_fkgs_tables.csv`
