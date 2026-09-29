# Báo cáo thực nghiệm FKG-E

Run ID: `tan_20260929_brset_fusion_kfold100_kb6_baseline_core_isolated`

Trạng thái: `completed`

Chế độ quick: `False`

## KB6

| ratio | n_rules_mean | fisa_sequential_avg_time_per_query_ms_mean | fisa_sequential_avg_time_per_query_ms_std | fisa_sequential_avg_time_per_query_ms_ci95_low | fisa_sequential_avg_time_per_query_ms_ci95_high | fisa_lookup_avg_time_per_query_ms_mean | fisa_lookup_avg_time_per_query_ms_std | fisa_lookup_avg_time_per_query_ms_ci95_low | fisa_lookup_avg_time_per_query_ms_ci95_high | fkge_avg_time_per_query_ms_mean | fkge_avg_time_per_query_ms_std | fkge_avg_time_per_query_ms_ci95_low | fkge_avg_time_per_query_ms_ci95_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2000 | 211.0000 | 1.0909 | 0.0866 | 1.0258 | 1.1566 | 0.0455 | 0.0005 | 0.0450 | 0.0459 | 0.1456 | 0.0283 | 0.1356 | 0.1570 |
| 0.4000 | 422.0000 | 2.0976 | 0.1274 | 1.9973 | 2.1954 | 0.0458 | 0.0006 | 0.0454 | 0.0463 | 0.1464 | 0.0196 | 0.1420 | 0.1539 |
| 0.6000 | 633.0000 | 3.1088 | 0.2043 | 2.9506 | 3.2671 | 0.0462 | 0.0005 | 0.0458 | 0.0466 | 0.1507 | 0.0016 | 0.1501 | 0.1515 |
| 0.8000 | 844.0000 | 4.1431 | 0.2635 | 3.9304 | 4.3436 | 0.0463 | 0.0003 | 0.0461 | 0.0465 | 0.1564 | 0.0025 | 0.1552 | 0.1575 |
| 1.0000 | 1054.6000 | 4.9030 | 0.2480 | 4.7100 | 5.0939 | 0.0452 | 0.0005 | 0.0448 | 0.0455 | 0.1628 | 0.0068 | 0.1598 | 0.1654 |

Hệ số góc log-log: fisa_sequential=0.9453, fisa_lookup=0.0017, fkge=0.0662. `*_std` là SD mẫu trên 5 fold (FISA) hoặc 25 lượt fold × seed (FKG-E); CI bootstrap theo fold.

## Baseline

| method | status | official_baseline | auc_roc_mean | auc_roc_std | auc_roc_ci95_low | auc_roc_ci95_high | f1_mean | f1_std | balanced_accuracy_mean | balanced_accuracy_std | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | accuracy_mean | agreement_mean | fidelity_balanced_accuracy_mean | cohen_kappa_mean | train_time_s_mean | avg_time_per_query_ms_mean | n_parameters_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FISA lookup | completed | True | 0.8231 | 0.0791 | 0.7568 | 0.8805 | 0.2845 | 0.0875 | 0.7141 | 0.0718 | 0.6649 | 0.7731 | 0.6376 |  |  |  | 0.0060 | 0.0456 |  |
| DeepWalk-lite + kNN | completed | False | 0.8471 | 0.0428 | 0.8226 | 0.8742 | 0.4625 | 0.0905 | 0.7300 | 0.0560 | 0.6963 | 0.7700 | 0.8999 |  |  |  | 6.2017 | 0.1179 | 2508.8000 |
| TransE-lite + kNN | completed | False | 0.7925 | 0.0650 | 0.7533 | 0.8432 | 0.4110 | 0.0913 | 0.6724 | 0.0464 | 0.6436 | 0.7013 | 0.9109 |  |  |  | 3.2958 | 0.1176 | 2508.8000 |
| DistMult-lite + kNN | completed | False | 0.8432 | 0.0424 | 0.8173 | 0.8742 | 0.4125 | 0.1301 | 0.6858 | 0.0792 | 0.6403 | 0.7440 | 0.9010 |  |  |  | 4.6544 | 0.1205 | 2508.8000 |
| MLP fuzzy features | completed | True | 0.9291 | 0.0246 | 0.9070 | 0.9488 | 0.5761 | 0.0459 | 0.8213 | 0.0542 | 0.7797 | 0.8624 | 0.9172 |  |  |  | 0.3929 | 0.0022 | 7066.6000 |
| FKG-E unsupervised | completed | True | 0.6206 | 0.2096 | 0.5889 | 0.6744 | 0.1625 | 0.1106 | 0.5621 | 0.0958 | 0.5371 | 0.5905 | 0.6745 | 0.5976 | 0.5480 | 0.1021 | 10.0446 | 0.1624 | 5017.6000 |
| FKG-E full | completed | True | 0.9305 | 0.0170 | 0.9149 | 0.9437 | 0.5142 | 0.0560 | 0.8411 | 0.0408 | 0.8069 | 0.8680 | 0.8744 | 0.7033 | 0.6685 | 0.3498 | 9.9752 | 0.1770 | 5017.6000 |
| Node2Vec + kNN | not_implemented_node2vec_bias_parameters | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| XGBoost fuzzy/original features | unavailable_dependency_xgboost | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TransE/DistMult (PyKEEN standard) | unavailable_dependency_pykeen | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

`*_std` là độ lệch chuẩn mẫu của 25 lượt fold × seed (FISA: 5 fold). Khoảng tin cậy 95% lấy mẫu lại theo 5 fold (trung bình seed trong từng fold). Các baseline hậu tố -lite chỉ dùng kiểm tra luồng, không là đối chứng chuẩn.
