# Báo cáo thực nghiệm FKG-E

Run ID: `tan_20260929_brset_fusion_kfold100_ablation`

Trạng thái: `completed`

Chế độ quick: `False`

## Ablation

| variant | auc_roc_mean | auc_roc_std | auc_roc_ci95_low | auc_roc_ci95_high | f1_mean | f1_std | balanced_accuracy_mean | balanced_accuracy_std | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | agreement_mean | agreement_std | mean_kl_divergence_mean | mean_kl_divergence_std | delta_auc_vs_full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No SGNS | 0.9306 | 0.0170 | 0.9150 | 0.9437 | 0.5128 | 0.0553 | 0.8407 | 0.0407 | 0.8068 | 0.8674 | 0.7033 | 0.1399 | 0.7204 | 0.0771 | 0.0001 |
| No node loss | 0.9305 | 0.0184 | 0.9137 | 0.9451 | 0.5136 | 0.0528 | 0.8369 | 0.0382 | 0.8104 | 0.8635 | 0.7027 | 0.1460 | 0.7554 | 0.0818 | -3.73e-05 |
| No FISA distillation | 0.9300 | 0.0155 | 0.9158 | 0.9416 | 0.5160 | 0.0473 | 0.8354 | 0.0437 | 0.8009 | 0.8672 | 0.6999 | 0.1421 | 1.5092 | 0.1645 | -0.0005 |
| No label prediction | 0.6206 | 0.2096 | 0.5889 | 0.6744 | 0.1625 | 0.1106 | 0.5621 | 0.0958 | 0.5371 | 0.5905 | 0.5976 | 0.1584 | 0.0002 | 4.25e-05 | -0.3100 |
| No L2 regularization | 0.9305 | 0.0170 | 0.9149 | 0.9437 | 0.5142 | 0.0560 | 0.8411 | 0.0408 | 0.8069 | 0.8680 | 0.7033 | 0.1402 | 0.7207 | 0.0764 | -1.85e-05 |
| Uniform mean pooling | 0.9231 | 0.0205 | 0.9081 | 0.9381 | 0.4884 | 0.0668 | 0.8445 | 0.0219 | 0.8305 | 0.8573 | 0.7122 | 0.1230 | 0.5845 | 0.0642 | -0.0075 |
| Prediction only | 0.9303 | 0.0176 | 0.9139 | 0.9433 | 0.5276 | 0.0432 | 0.8366 | 0.0412 | 0.8068 | 0.8664 | 0.6984 | 0.1461 | 1.5917 | 0.1897 | -0.0002 |
| SGNS only | 0.5980 | 0.1932 | 0.5585 | 0.6541 | 0.1307 | 0.0872 | 0.5302 | 0.0785 | 0.5070 | 0.5607 | 0.5519 | 0.1694 | 0.0002 | 4.64e-05 | -0.3325 |
| Full FKG-E | 0.9305 | 0.0170 | 0.9149 | 0.9437 | 0.5142 | 0.0560 | 0.8411 | 0.0408 | 0.8069 | 0.8680 | 0.7033 | 0.1402 | 0.7206 | 0.0764 | 0.0000 |

Chưa triển khai: `L_edge, L_A, L_B, L_rule, attention_pooling`. `*_std` là SD mẫu trên 5 fold × 5 seed; CI bootstrap theo fold.
