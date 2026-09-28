# Báo cáo thực nghiệm FKG-E

Run ID: `tan_20260928_brset_fusion_quick`  

Trạng thái: `completed`  

Chế độ quick: `True`

## KB1

| dataset | method | n_rules | auc_roc_mean | f1_mean | accuracy_mean | balanced_accuracy_mean | agreement_mean | mean_kl_divergence_mean | avg_time_per_query_ms_mean | train_time_s_mean | synthetic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BRSET fusion | FISA sequential | 1054.6000 | 0.8231 | 0.2845 | 0.6376 | 0.7141 |  |  | 5.2661 | 0.0070 | False |
| BRSET fusion | FISA lookup | 1054.6000 | 0.8231 | 0.2845 | 0.6376 | 0.7141 |  |  | 0.0466 | 0.0068 | False |
| BRSET fusion | FKG-E unsupervised | 1054.6000 | 0.6485 | 0.0000 | 0.9197 | 0.5000 | 0.5889 | 0.0249 | 0.1008 | 60.7870 | False |
| BRSET fusion | FKG-E full | 1054.6000 | 0.9214 | 0.4984 | 0.8502 | 0.8533 | 0.7162 | 0.2024 | 0.1095 | 56.4702 | False |

## KB2

| dataset | method | n_rules | auc_roc_mean | balanced_accuracy_mean | avg_time_per_query_ms_mean |
|---|---|---|---|---|---|
| BRSET fusion / FKG | FISA sequential | 1054.6000 | 0.8231 | 0.7141 | 5.1184 |
| BRSET fusion / FKG | FISA lookup | 1054.6000 | 0.8231 | 0.7141 | 0.0454 |
| BRSET fusion / FKG | FKG-E unsupervised | 1054.6000 | 0.6485 | 0.5000 | 0.0995 |
| BRSET fusion / FKG | FKG-E full | 1054.6000 | 0.9214 | 0.8533 | 0.1036 |
| BRSET fusion / FKGS 30% | FISA sequential | 316.6000 | 0.7957 | 0.6017 | 1.5528 |
| BRSET fusion / FKGS 30% | FISA lookup | 316.6000 | 0.7957 | 0.6017 | 0.0459 |
| BRSET fusion / FKGS 30% | FKG-E unsupervised | 316.6000 | 0.5938 | 0.5000 | 0.0816 |
| BRSET fusion / FKGS 30% | FKG-E full | 316.6000 | 0.8800 | 0.5195 | 0.0808 |

Kết luận tự động: `triet_tieu_hoac_khong_ro`; delta AUC=-0.0414, tỉ số thời gian=1.2815.

## KB3

| d | auc_roc_mean | auc_pr_mean | n_parameters_mean | train_time_s_mean | avg_time_per_query_ms_mean | embedding_memory_bytes_mean |
|---|---|---|---|---|---|---|
| 8 | 0.9324 | 0.6253 | 1264.0000 | 55.3410 | 0.0922 | 10112.0000 |
| 32 | 0.9347 | 0.6159 | 5056.0000 | 59.2789 | 0.1049 | 40448.0000 |

Chọn `d*=8`.

## KB4

| weight | multiplier | effective_value | auc_roc_mean | agreement_mean | mean_kl_divergence_mean |
|---|---|---|---|---|---|
| lambda_S | 0.0000 | 0.0000 | 0.9347 | 0.8182 | 0.2072 |
| lambda_S | 1.0000 | 1.0000 | 0.9347 | 0.8058 | 0.2069 |
| lambda_N | 0.0000 | 0.0000 | 0.9327 | 0.7975 | 0.2252 |
| lambda_N | 1.0000 | 1.0000 | 0.9347 | 0.8058 | 0.2069 |
| lambda_I | 0.0000 | 0.0000 | 0.9342 | 0.8099 | 0.2244 |
| lambda_I | 1.0000 | 2.0000 | 0.9347 | 0.8058 | 0.2069 |
| lambda_P | 0.0000 | 0.0000 | 0.6998 | 0.6529 | 0.0273 |
| lambda_P | 1.0000 | 20.0000 | 0.9347 | 0.8058 | 0.2069 |
| lambda_C | 0.0000 | 0.0000 | 0.9347 | 0.8058 | 0.2070 |
| lambda_C | 1.0000 | 0.0000 | 0.9347 | 0.8058 | 0.2069 |

Chưa triển khai trong model hiện tại: `lambda_E, lambda_A, lambda_B, lambda_R`.

## KB5

| w | K | auc_roc_mean | auc_pr_mean | agreement_mean |
|---|---|---|---|---|
| 1 | 2 | 0.9234 | 0.5553 | 0.8017 |
| 1 | 5 | 0.9234 | 0.5553 | 0.8017 |
| 2 | 2 | 0.9347 | 0.6159 | 0.8058 |
| 2 | 5 | 0.9347 | 0.6159 | 0.8058 |

Biên độ AUC=0.0113; H-E5=ủng hộ

## KB6

| ratio | n_rules_mean | fisa_sequential_avg_time_per_query_ms_mean | fisa_lookup_avg_time_per_query_ms_mean | fkge_avg_time_per_query_ms_mean |
|---|---|---|---|---|
| 0.4000 | 426.0000 | 2.1294 | 0.0479 | 0.0860 |
| 1.0000 | 1064.0000 | 5.4794 | 0.0452 | 0.1047 |

Hệ số góc log-log: fisa_sequential=1.0326, fisa_lookup=-0.0618, fkge=0.2153

## Ablation

| variant | auc_roc_mean | f1_mean | balanced_accuracy_mean | agreement_mean | mean_kl_divergence_mean | delta_auc_vs_full |
|---|---|---|---|---|---|---|
| No SGNS | 0.9347 | 0.4146 | 0.8236 | 0.8182 | 0.2072 | 0.0000 |
| No node loss | 0.9327 | 0.4658 | 0.8439 | 0.7975 | 0.2252 | -0.0020 |
| No FISA distillation | 0.9342 | 0.4186 | 0.8419 | 0.8099 | 0.2244 | -0.0005 |
| No label prediction | 0.6998 | 0.0000 | 0.5000 | 0.6529 | 0.0273 | -0.2349 |
| No L2 regularization | 0.9347 | 0.4000 | 0.8169 | 0.8058 | 0.2070 | 0.0000 |
| Uniform mean pooling | 0.9275 | 0.0000 | 0.5000 | 0.6529 | 0.0253 | -0.0072 |
| Prediction only | 0.9331 | 0.4474 | 0.8372 | 0.8099 | 0.2424 | -0.0016 |
| SGNS only | 0.7311 | 0.0000 | 0.5000 | 0.6529 | 0.0285 | -0.2036 |
| Full FKG-E | 0.9347 | 0.4000 | 0.8169 | 0.8058 | 0.2069 | 0.0000 |

Chưa triển khai: `L_edge, L_A, L_B, L_rule, attention_pooling`.

## Baseline

| method | status | official_baseline | auc_roc_mean | f1_mean | accuracy_mean | train_time_s_mean | avg_time_per_query_ms_mean | n_parameters_mean |
|---|---|---|---|---|---|---|---|---|
| FISA lookup | completed | True | 0.9175 | 0.3654 | 0.7273 | 0.0061 | 0.0449 |  |
| DeepWalk-lite + kNN | completed | False | 0.8732 | 0.4762 | 0.9091 | 5.7030 | 0.1127 | 2528.0000 |
| TransE-lite + kNN | completed | False | 0.8857 | 0.4762 | 0.9091 | 3.3195 | 0.1134 | 2528.0000 |
| DistMult-lite + kNN | completed | False | 0.8992 | 0.6047 | 0.9298 | 4.2934 | 0.1147 | 2528.0000 |
| MLP fuzzy features | completed | True | 0.9644 | 0.6800 | 0.9339 | 0.4195 | 0.0023 | 7105.0000 |
| FKG-E unsupervised | completed | True | 0.6998 | 0.0000 | 0.9174 | 54.4036 | 0.0990 | 5056.0000 |
| FKG-E full | completed | True | 0.9347 | 0.4000 | 0.7893 | 54.4627 | 0.0995 | 5056.0000 |
| Node2Vec + kNN | not_implemented_node2vec_bias_parameters | False |  |  |  |  |  |  |
| XGBoost fuzzy/original features | unavailable_dependency_xgboost | False |  |  |  |  |  |  |
| TransE/DistMult (PyKEEN standard) | unavailable_dependency_pykeen | False |  |  |  |  |  |  |
