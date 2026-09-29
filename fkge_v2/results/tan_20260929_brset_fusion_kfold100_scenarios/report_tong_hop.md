# Báo cáo thực nghiệm FKG-E

Run ID: `tan_20260929_brset_fusion_kfold100_scenarios`

Trạng thái: `partial_completed_kb3_kb5`

Chế độ quick: `False`

## KB3

| d | auc_roc_mean | auc_roc_std | auc_roc_ci95_low | auc_roc_ci95_high | auc_pr_mean | auc_pr_std | balanced_accuracy_mean | balanced_accuracy_std | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | n_parameters_mean | train_time_s_mean | avg_time_per_query_ms_mean | embedding_memory_bytes_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.9306 | 0.0170 | 0.9153 | 0.9445 | 0.6059 | 0.0553 | 0.8410 | 0.0372 | 0.8134 | 0.8659 | 1254.4000 | 9.1992 | 0.1805 | 10035.2000 |
| 16 | 0.9306 | 0.0169 | 0.9154 | 0.9442 | 0.6077 | 0.0557 | 0.8401 | 0.0391 | 0.8108 | 0.8656 | 2508.8000 | 8.6352 | 0.1588 | 20070.4000 |
| 32 | 0.9305 | 0.0170 | 0.9149 | 0.9437 | 0.6085 | 0.0546 | 0.8411 | 0.0408 | 0.8069 | 0.8680 | 5017.6000 | 10.0147 | 0.1671 | 40140.8000 |
| 64 | 0.9308 | 0.0171 | 0.9150 | 0.9436 | 0.6112 | 0.0539 | 0.8425 | 0.0409 | 0.8097 | 0.8697 | 10035.2000 | 12.9465 | 0.1868 | 80281.6000 |
| 128 | 0.9313 | 0.0172 | 0.9154 | 0.9438 | 0.6137 | 0.0506 | 0.8400 | 0.0403 | 0.8077 | 0.8696 | 20070.4000 | 22.1546 | 0.1942 | 160563.2000 |

Chọn `d*=8`. `*_std` là SD mẫu trên 5 fold × 5 seed; CI bootstrap theo fold sau khi lấy trung bình seed.

## KB5

| cooccurrence | K | auc_roc_mean | auc_roc_std | auc_roc_ci95_low | auc_roc_ci95_high | auc_pr_mean | auc_pr_std | agreement_mean | agreement_std |
|---|---|---|---|---|---|---|---|---|---|
| full_rule | 2 | 0.9305 | 0.0169 | 0.9150 | 0.9435 | 0.6093 | 0.0543 | 0.7027 | 0.1405 |
| full_rule | 5 | 0.9305 | 0.0170 | 0.9149 | 0.9437 | 0.6085 | 0.0546 | 0.7033 | 0.1402 |
| full_rule | 10 | 0.9306 | 0.0170 | 0.9150 | 0.9438 | 0.6090 | 0.0543 | 0.7032 | 0.1407 |
| window_2 | 2 | 0.9328 | 0.0191 | 0.9158 | 0.9482 | 0.6196 | 0.0614 | 0.7000 | 0.1450 |
| window_2 | 5 | 0.9328 | 0.0191 | 0.9159 | 0.9482 | 0.6193 | 0.0612 | 0.7005 | 0.1450 |
| window_2 | 10 | 0.9327 | 0.0191 | 0.9158 | 0.9482 | 0.6191 | 0.0616 | 0.6999 | 0.1452 |

Biên độ AUC=0.0023. Đây là thống kê mô tả; chưa kết luận H-E5 khi đóng góp của SGNS chưa được xác nhận. `*_std` là SD mẫu trên 5 fold × 5 seed; CI bootstrap theo fold.
