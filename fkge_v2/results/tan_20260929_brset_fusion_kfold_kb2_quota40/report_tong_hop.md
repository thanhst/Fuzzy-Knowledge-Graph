# Báo cáo thực nghiệm FKG-E

Run ID: `tan_20260929_brset_fusion_kfold_kb2_quota40`

Trạng thái: `completed`

Chế độ quick: `False`

## KB2

| dataset | method | n_rules | auc_roc_mean | auc_roc_std | auc_roc_ci95_low | auc_roc_ci95_high | balanced_accuracy_mean | balanced_accuracy_std | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | agreement_mean | agreement_std | agreement_ci95_low | agreement_ci95_high | fidelity_balanced_accuracy_mean | fidelity_balanced_accuracy_std | avg_time_per_query_ms_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BRSET fusion / FKG | FISA sequential | 1054.6000 | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.7141 | 0.0718 | 0.6649 | 0.7731 |  |  |  |  |  |  | 5.5502 |
| BRSET fusion / FKG | FISA lookup | 1054.6000 | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.7141 | 0.0718 | 0.6649 | 0.7731 |  |  |  |  |  |  | 0.0499 |
| BRSET fusion / FKG | FKG-E unsupervised | 1054.6000 | 0.6055 | 0.1904 | 0.5674 | 0.6597 | 0.5458 | 0.1024 | 0.5168 | 0.5748 | 0.5709 | 0.1631 | 0.5281 | 0.6138 | 0.5446 | 0.0914 | 0.1767 |
| BRSET fusion / FKG | FKG-E full | 1054.6000 | 0.9114 | 0.0312 | 0.8866 | 0.9370 | 0.8223 | 0.0499 | 0.7953 | 0.8681 | 0.7159 | 0.1371 | 0.5996 | 0.8322 | 0.6927 | 0.0878 | 0.1676 |
| BRSET fusion / FKGS 30% | FISA sequential | 316.6000 | 0.7693 | 0.0992 | 0.6943 | 0.8455 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |  |  |  |  |  |  | 1.5278 |
| BRSET fusion / FKGS 30% | FISA lookup | 316.6000 | 0.7693 | 0.0992 | 0.6943 | 0.8455 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |  |  |  |  |  |  | 0.0463 |
| BRSET fusion / FKGS 30% | FKG-E unsupervised | 316.6000 | 0.6040 | 0.1856 | 0.5675 | 0.6571 | 0.5318 | 0.0997 | 0.5053 | 0.5559 | 0.4938 | 0.2965 | 0.4418 | 0.5354 |  |  | 0.1417 |
| BRSET fusion / FKGS 30% | FKG-E full | 316.6000 | 0.9150 | 0.0308 | 0.8897 | 0.9402 | 0.8207 | 0.0610 | 0.7853 | 0.8716 | 0.1817 | 0.0503 | 0.1463 | 0.2169 |  |  | 0.1420 |

`*_std` là độ lệch chuẩn mẫu của 25 lượt fold × seed (FISA: 5 fold). Trạng thái đánh giá: `fisa_teacher_collapsed_fidelity_unresolved`; delta AUC=0.0036, delta BalAcc=-0.0016, tỉ số thời gian=1.1803. Tập con 30% dùng hạn ngạch tối thiểu 40% luật cho mỗi lớp; đây là mô phỏng lấy mẫu, không phải FKGS chính thức.
