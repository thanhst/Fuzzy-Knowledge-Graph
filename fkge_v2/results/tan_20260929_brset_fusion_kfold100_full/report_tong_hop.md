# Báo cáo thực nghiệm FKG-E

Run ID: `tan_20260929_brset_fusion_kfold100_full`

Trạng thái: `partial_completed_kb1_kb2`

Chế độ quick: `False`

## KB1

| dataset | method | n_rules | auc_roc_mean | auc_roc_std | auc_roc_ci95_low | auc_roc_ci95_high | f1_mean | f1_std | accuracy_mean | balanced_accuracy_mean | balanced_accuracy_std | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | agreement_mean | agreement_std | agreement_ci95_low | agreement_ci95_high | fidelity_balanced_accuracy_mean | fidelity_balanced_accuracy_std | cohen_kappa_mean | fidelity_bound_coverage_mean | mean_kl_divergence_mean | avg_time_per_query_ms_mean | train_time_s_mean | synthetic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BRSET fusion | FISA sequential | 1054.6000 | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.2845 | 0.0875 | 0.6376 | 0.7141 | 0.0718 | 0.6649 | 0.7731 |  |  |  |  |  |  |  |  |  | 5.8488 | 0.0065 | False |
| BRSET fusion | FISA lookup | 1054.6000 | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.2845 | 0.0875 | 0.6376 | 0.7141 | 0.0718 | 0.6649 | 0.7731 |  |  |  |  |  |  |  |  |  | 0.0535 | 0.0062 | False |
| BRSET fusion | FKG-E unsupervised | 1054.6000 | 0.6206 | 0.2096 | 0.5889 | 0.6744 | 0.1625 | 0.1106 | 0.6745 | 0.5621 | 0.0958 | 0.5371 | 0.5905 | 0.5976 | 0.1584 | 0.5504 | 0.6448 | 0.5480 | 0.0740 | 0.1021 | 0.5620 | 0.0002 | 0.2044 | 11.4703 | False |
| BRSET fusion | FKG-E full | 1054.6000 | 0.9305 | 0.0170 | 0.9149 | 0.9437 | 0.5142 | 0.0560 | 0.8744 | 0.8411 | 0.0408 | 0.8069 | 0.8680 | 0.7033 | 0.1402 | 0.5889 | 0.8178 | 0.6685 | 0.0686 | 0.3498 | 0.0046 | 0.7206 | 0.1872 | 11.1534 | False |

`*_std` là độ lệch chuẩn mẫu của 25 lượt fold × seed (FISA: 5 fold). Khoảng tin cậy 95%: bootstrap theo 5 fold, lấy trung bình seed trong từng fold (10.000 lượt lấy mẫu). Objective hiện chỉ gồm L_SGNS, L_node, L_inf, L_pred và L2; chưa có L_edge, L_A, L_B, L_rule. FISA và ngưỡng quyết định chưa được hiệu chỉnh trên tập xác thực lồng theo bệnh nhân.

## KB2

| dataset | method | n_rules | auc_roc_mean | auc_roc_std | auc_roc_ci95_low | auc_roc_ci95_high | balanced_accuracy_mean | balanced_accuracy_std | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | agreement_mean | agreement_std | agreement_ci95_low | agreement_ci95_high | fidelity_balanced_accuracy_mean | fidelity_balanced_accuracy_std | avg_time_per_query_ms_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BRSET fusion / FKG | FISA sequential | 1054.6000 | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.7141 | 0.0718 | 0.6649 | 0.7731 |  |  |  |  |  |  | 5.4889 |
| BRSET fusion / FKG | FISA lookup | 1054.6000 | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.7141 | 0.0718 | 0.6649 | 0.7731 |  |  |  |  |  |  | 0.0510 |
| BRSET fusion / FKG | FKG-E unsupervised | 1054.6000 | 0.6206 | 0.2096 | 0.5889 | 0.6744 | 0.5621 | 0.0958 | 0.5371 | 0.5905 | 0.5976 | 0.1584 | 0.5504 | 0.6448 | 0.5480 | 0.0740 | 0.1749 |
| BRSET fusion / FKG | FKG-E full | 1054.6000 | 0.9305 | 0.0170 | 0.9149 | 0.9437 | 0.8411 | 0.0408 | 0.8069 | 0.8680 | 0.7033 | 0.1402 | 0.5889 | 0.8178 | 0.6685 | 0.0686 | 0.1904 |
| BRSET fusion / FKGS 30% | FISA sequential | 316.6000 | 0.7693 | 0.0992 | 0.6943 | 0.8455 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |  |  |  |  |  |  | 1.7733 |
| BRSET fusion / FKGS 30% | FISA lookup | 316.6000 | 0.7693 | 0.0992 | 0.6943 | 0.8455 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |  |  |  |  |  |  | 0.0588 |
| BRSET fusion / FKGS 30% | FKG-E unsupervised | 316.6000 | 0.5302 | 0.1541 | 0.4892 | 0.5665 | 0.5140 | 0.0767 | 0.4908 | 0.5368 | 0.8467 | 0.2627 | 0.7339 | 0.9595 |  |  | 0.1853 |
| BRSET fusion / FKGS 30% | FKG-E full | 316.6000 | 0.9333 | 0.0184 | 0.9166 | 0.9482 | 0.8439 | 0.0503 | 0.8074 | 0.8797 | 0.1751 | 0.0474 | 0.1393 | 0.2091 |  |  | 0.1829 |

`*_std` là độ lệch chuẩn mẫu của 25 lượt fold × seed (FISA: 5 fold). Trạng thái đánh giá: `fisa_teacher_collapsed_fidelity_unresolved`; delta AUC=0.0028, delta BalAcc=0.0028, tỉ số thời gian=1.0411. Tập con 30% dùng hạn ngạch tối thiểu 40% luật cho mỗi lớp; đây là mô phỏng lấy mẫu, không phải FKGS chính thức.
