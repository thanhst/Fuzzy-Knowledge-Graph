# Báo cáo thực nghiệm FKG-E

Run ID: `tan_20260929_brset_fusion_kfold_revised`

Trạng thái: `completed`

Chế độ quick: `False`

## KB1

| dataset | method | n_rules | auc_roc_mean | auc_roc_std | auc_roc_ci95_low | auc_roc_ci95_high | f1_mean | f1_std | accuracy_mean | balanced_accuracy_mean | balanced_accuracy_std | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | agreement_mean | agreement_std | agreement_ci95_low | agreement_ci95_high | fidelity_balanced_accuracy_mean | fidelity_balanced_accuracy_std | cohen_kappa_mean | fidelity_bound_coverage_mean | mean_kl_divergence_mean | avg_time_per_query_ms_mean | train_time_s_mean | synthetic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BRSET fusion | FISA sequential | 1054.6000 | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.2845 | 0.0875 | 0.6376 | 0.7141 | 0.0718 | 0.6649 | 0.7731 |  |  |  |  |  |  |  |  |  | 5.4913 | 0.0064 | False |
| BRSET fusion | FISA lookup | 1054.6000 | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.2845 | 0.0875 | 0.6376 | 0.7141 | 0.0718 | 0.6649 | 0.7731 |  |  |  |  |  |  |  |  |  | 0.0485 | 0.0059 | False |
| BRSET fusion | FKG-E unsupervised | 1054.6000 | 0.6055 | 0.1904 | 0.5674 | 0.6597 | 0.1607 | 0.0912 | 0.6454 | 0.5458 | 0.1024 | 0.5168 | 0.5748 | 0.5709 | 0.1631 | 0.5281 | 0.6138 | 0.5446 | 0.0914 | 0.0956 | 0.5436 | 0.0002 | 0.1653 | 21.7081 | False |
| BRSET fusion | FKG-E full | 1054.6000 | 0.9114 | 0.0312 | 0.8866 | 0.9370 | 0.4761 | 0.0898 | 0.8542 | 0.8223 | 0.0499 | 0.7953 | 0.8681 | 0.7159 | 0.1371 | 0.5996 | 0.8322 | 0.6927 | 0.0878 | 0.3852 | 0.0088 | 0.1309 | 0.1706 | 21.5204 | False |

`*_std` là độ lệch chuẩn mẫu của 25 lượt fold × seed (FISA: 5 fold). Khoảng tin cậy 95%: bootstrap theo 5 fold, lấy trung bình seed trong từng fold (10.000 lượt lấy mẫu). Objective hiện chỉ gồm L_SGNS, L_node, L_inf, L_pred và L2; chưa có L_edge, L_A, L_B, L_rule. FISA và ngưỡng quyết định chưa được hiệu chỉnh trên tập xác thực lồng theo bệnh nhân.

## Baseline

| method | status | official_baseline | auc_roc_mean | auc_roc_std | auc_roc_ci95_low | auc_roc_ci95_high | f1_mean | f1_std | balanced_accuracy_mean | balanced_accuracy_std | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | accuracy_mean | agreement_mean | fidelity_balanced_accuracy_mean | cohen_kappa_mean | train_time_s_mean | avg_time_per_query_ms_mean | n_parameters_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FISA lookup | completed | True | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.2845 | 0.0875 | 0.7141 | 0.0718 | 0.6649 | 0.7731 | 0.6376 |  |  |  | 0.0059 | 0.0456 |  |
| DeepWalk-lite + kNN | completed | False | 0.8582 | 0.0566 | 0.8212 | 0.8920 | 0.4656 | 0.0843 | 0.7340 | 0.0595 | 0.7055 | 0.7619 | 0.9019 |  |  |  | 6.3988 | 0.1268 | 2508.8000 |
| TransE-lite + kNN | completed | False | 0.7666 | 0.0631 | 0.7326 | 0.8182 | 0.3718 | 0.0843 | 0.6545 | 0.0489 | 0.6266 | 0.6841 | 0.9051 |  |  |  | 3.5154 | 0.1233 | 2508.8000 |
| DistMult-lite + kNN | completed | False | 0.8456 | 0.0421 | 0.8214 | 0.8761 | 0.4160 | 0.0988 | 0.6911 | 0.0647 | 0.6582 | 0.7404 | 0.8999 |  |  |  | 4.9310 | 0.1302 | 2508.8000 |
| MLP fuzzy features | completed | True | 0.9291 | 0.0246 | 0.9070 | 0.9488 | 0.5761 | 0.0459 | 0.8213 | 0.0542 | 0.7797 | 0.8624 | 0.9172 |  |  |  | 0.4555 | 0.0025 | 7066.6000 |
| FKG-E unsupervised | completed | True | 0.6055 | 0.1904 | 0.5674 | 0.6597 | 0.1607 | 0.0912 | 0.5458 | 0.1024 | 0.5168 | 0.5748 | 0.6454 | 0.5709 | 0.5446 | 0.0956 | 23.5653 | 0.1823 | 5017.6000 |
| FKG-E full | completed | True | 0.9114 | 0.0312 | 0.8866 | 0.9370 | 0.4761 | 0.0898 | 0.8223 | 0.0499 | 0.7953 | 0.8681 | 0.8542 | 0.7159 | 0.6927 | 0.3852 | 24.0986 | 0.1756 | 5017.6000 |
| Node2Vec + kNN | not_implemented_node2vec_bias_parameters | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| XGBoost fuzzy/original features | unavailable_dependency_xgboost | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TransE/DistMult (PyKEEN standard) | unavailable_dependency_pykeen | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

`*_std` là độ lệch chuẩn mẫu của 25 lượt fold × seed (FISA: 5 fold). Khoảng tin cậy 95% lấy mẫu lại theo 5 fold (trung bình seed trong từng fold). Các baseline hậu tố -lite chỉ dùng kiểm tra luồng, không là đối chứng chuẩn.
