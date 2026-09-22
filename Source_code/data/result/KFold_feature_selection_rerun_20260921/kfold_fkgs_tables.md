# KFold FKGS summary tables

Note: KFold uses 5 folds. Timing starts from precomputed feature CSVs, so it does not include raw image preprocessing/feature extraction. Training time is fold preprocessing over those CSVs + FIS + FKGS sampling/train. Test time is FKGS test time. Total time is training + test.
Selected feature counts are stored in selected_feature_count.
Sensitivity, Specificity and F1 use the positive diabetic-retinopathy class. Values are mean +/- sample standard deviation across the 5 validation folds.

## Bang 3.2. Phuong phap lua chon thuoc tinh voi ti le mau 15% va nguong sai so 0.2

| Mo hinh | Mo thuc | Acc (%) | Sensitivity (%) | Specificity (%) | F1 (%) | AUC-ROC (%) | AUC-PR (%) | Thoi gian huan luyen | Thoi gian kiem tra | Tong thoi gian (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FKG-UM | Du lieu dang bang full | 91.06 +/- 1.04 | 21.42 +/- 17.43 | 97.12 +/- 2.26 | 24.04 +/- 17.93 | 72.21 +/- 8.38 | 24.27 +/- 13.04 | 3.55 | 6.81 | 10.36 |
| FKG-UM | Du lieu anh | 70.62 +/- 10.81 | 36.79 +/- 25.92 | 73.53 +/- 13.68 | 15.14 +/- 7.92 | 55.56 +/- 9.63 | 11.63 +/- 3.78 | 2.52 | 4.70 | 7.22 |
| FKG-MM | Du lieu anh+bang | 91.64 +/- 0.44 | 6.21 +/- 5.79 | 99.10 +/- 0.63 | 10.16 +/- 8.26 | 65.38 +/- 7.24 | 14.43 +/- 3.17 | 5.35 | 11.24 | 16.58 |
| FKG-MM | Fusion Filter | 91.56 +/- 0.61 | 3.11 +/- 2.84 | 99.28 +/- 0.68 | 5.48 +/- 5.04 | 61.65 +/- 4.46 | 12.07 +/- 3.10 | 5.38 | 12.45 | 17.82 |
| FKG-MM | Fusion Hadamard | 76.07 +/- 7.21 | 30.95 +/- 16.25 | 80.02 +/- 8.47 | 16.85 +/- 8.64 | 57.50 +/- 7.65 | 11.78 +/- 2.99 | 2.47 | 4.38 | 6.84 |
| FKG-MM | Fusion Tensor | 91.64 +/- 0.56 | 0.00 +/- 0.00 | 99.64 +/- 0.59 | 0.00 +/- 0.00 | 50.22 +/- 0.42 | 8.09 +/- 0.24 | 57.55 | 43.53 | 101.08 |
| FKG-MM | Fusion Wrapper | 80.14 +/- 7.44 | 38.00 +/- 14.97 | 83.82 +/- 7.99 | 25.19 +/- 12.73 | 65.87 +/- 12.53 | 20.89 +/- 10.62 | 28.03 | 3.75 | 31.79 |

## Bang 3.3. Phuong phap lua chon thuoc tinh voi ti le mau 15% va nguong sai so 0.3

| Mo hinh | Mo thuc | Acc (%) | Sensitivity (%) | Specificity (%) | F1 (%) | AUC-ROC (%) | AUC-PR (%) | Thoi gian huan luyen | Thoi gian kiem tra | Tong thoi gian (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FKG-UM | Du lieu dang bang full | 90.48 +/- 1.76 | 23.79 +/- 24.54 | 96.31 +/- 3.44 | 23.80 +/- 18.22 | 73.60 +/- 7.69 | 24.47 +/- 12.57 | 3.56 | 6.85 | 10.40 |
| FKG-UM | Du lieu anh | 65.15 +/- 3.92 | 39.05 +/- 19.30 | 67.41 +/- 5.03 | 14.90 +/- 6.26 | 56.66 +/- 11.41 | 12.76 +/- 5.06 | 2.52 | 4.63 | 7.16 |
| FKG-MM | Du lieu anh+bang | 91.56 +/- 0.47 | 2.05 +/- 2.81 | 99.37 +/- 0.68 | 3.57 +/- 4.91 | 60.67 +/- 6.78 | 11.06 +/- 2.53 | 5.36 | 11.09 | 16.45 |
| FKG-MM | Fusion Filter | 91.64 +/- 0.33 | 5.16 +/- 5.14 | 99.19 +/- 0.49 | 8.55 +/- 8.52 | 58.33 +/- 2.93 | 11.59 +/- 3.16 | 5.42 | 12.24 | 17.65 |
| FKG-MM | Fusion Hadamard | 74.09 +/- 7.77 | 32.89 +/- 13.84 | 77.67 +/- 9.38 | 16.44 +/- 4.56 | 57.83 +/- 5.27 | 11.20 +/- 2.17 | 2.47 | 4.61 | 7.07 |
| FKG-MM | Fusion Tensor | 92.05 +/- 0.20 | 1.00 +/- 2.24 | 100.00 +/- 0.00 | 1.90 +/- 4.26 | 50.90 +/- 1.10 | 9.01 +/- 2.18 | 57.90 | 49.24 | 107.14 |
| FKG-MM | Fusion Wrapper | 78.40 +/- 7.10 | 28.79 +/- 10.56 | 82.73 +/- 7.70 | 18.83 +/- 9.21 | 59.33 +/- 11.12 | 18.30 +/- 7.91 | 28.03 | 3.71 | 31.74 |

## Bang 3.4. Phuong phap lua chon thuoc tinh voi ti le mau 20% va nguong sai so 0.2

| Mo hinh | Mo thuc | Acc (%) | Sensitivity (%) | Specificity (%) | F1 (%) | AUC-ROC (%) | AUC-PR (%) | Thoi gian huan luyen | Thoi gian kiem tra | Tong thoi gian (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FKG-UM | Du lieu dang bang full | 90.07 +/- 2.41 | 27.00 +/- 24.63 | 95.59 +/- 2.15 | 26.92 +/- 22.59 | 75.13 +/- 9.37 | 23.52 +/- 10.56 | 4.07 | 7.93 | 12.00 |
| FKG-UM | Du lieu anh | 66.23 +/- 4.99 | 37.95 +/- 19.24 | 68.68 +/- 5.78 | 15.12 +/- 6.80 | 53.90 +/- 14.27 | 12.91 +/- 3.67 | 2.58 | 4.82 | 7.40 |
| FKG-MM | Du lieu anh+bang | 91.64 +/- 0.62 | 6.21 +/- 8.67 | 99.10 +/- 0.45 | 9.79 +/- 12.58 | 69.41 +/- 5.75 | 15.30 +/- 2.68 | 6.26 | 13.83 | 20.09 |
| FKG-MM | Fusion Filter | 91.89 +/- 0.48 | 4.16 +/- 2.33 | 99.55 +/- 0.32 | 7.54 +/- 4.23 | 65.08 +/- 3.09 | 13.07 +/- 1.56 | 6.52 | 15.82 | 22.33 |
| FKG-MM | Fusion Hadamard | 68.63 +/- 14.19 | 39.00 +/- 11.11 | 71.18 +/- 16.05 | 17.59 +/- 4.07 | 58.18 +/- 5.93 | 13.23 +/- 3.84 | 2.51 | 4.89 | 7.40 |
| FKG-MM | Fusion Tensor | 91.72 +/- 0.30 | 0.00 +/- 0.00 | 99.73 +/- 0.25 | 0.00 +/- 0.00 | 50.36 +/- 0.41 | 8.10 +/- 0.26 | 78.27 | 52.44 | 130.71 |
| FKG-MM | Fusion Wrapper | 80.80 +/- 5.79 | 40.89 +/- 21.05 | 84.25 +/- 6.82 | 25.57 +/- 10.95 | 63.91 +/- 12.21 | 21.64 +/- 9.49 | 28.07 | 3.73 | 31.80 |

## Bang 3.5. Phuong phap lua chon thuoc tinh voi ti le mau 20% va nguong sai so 0.3

| Mo hinh | Mo thuc | Acc (%) | Sensitivity (%) | Specificity (%) | F1 (%) | AUC-ROC (%) | AUC-PR (%) | Thoi gian huan luyen | Thoi gian kiem tra | Tong thoi gian (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FKG-UM | Du lieu dang bang full | 90.73 +/- 1.59 | 24.68 +/- 19.28 | 96.49 +/- 2.83 | 26.65 +/- 14.72 | 71.36 +/- 8.26 | 22.92 +/- 12.31 | 4.07 | 7.89 | 11.96 |
| FKG-UM | Du lieu anh | 70.12 +/- 5.99 | 35.89 +/- 20.02 | 73.08 +/- 7.75 | 15.25 +/- 7.39 | 56.19 +/- 11.11 | 11.89 +/- 3.54 | 2.59 | 4.89 | 7.48 |
| FKG-MM | Du lieu anh+bang | 91.39 +/- 0.44 | 6.21 +/- 5.79 | 98.83 +/- 0.51 | 9.92 +/- 8.30 | 66.27 +/- 3.76 | 13.45 +/- 2.08 | 6.29 | 13.99 | 20.28 |
| FKG-MM | Fusion Filter | 91.72 +/- 0.51 | 6.21 +/- 5.79 | 99.19 +/- 0.66 | 10.17 +/- 8.27 | 62.64 +/- 2.11 | 12.62 +/- 1.52 | 6.57 | 15.46 | 22.02 |
| FKG-MM | Fusion Hadamard | 67.73 +/- 14.19 | 34.00 +/- 12.00 | 70.65 +/- 15.50 | 15.55 +/- 6.11 | 51.89 +/- 10.19 | 10.98 +/- 3.97 | 2.50 | 4.55 | 7.04 |
| FKG-MM | Fusion Tensor | 91.72 +/- 0.30 | 0.00 +/- 0.00 | 99.73 +/- 0.25 | 0.00 +/- 0.00 | 50.36 +/- 0.34 | 8.10 +/- 0.24 | 78.38 | 56.48 | 134.86 |
| FKG-MM | Fusion Wrapper | 81.13 +/- 6.36 | 42.00 +/- 16.49 | 84.53 +/- 7.13 | 27.50 +/- 11.29 | 64.71 +/- 12.66 | 21.82 +/- 10.02 | 28.08 | 3.87 | 31.95 |
