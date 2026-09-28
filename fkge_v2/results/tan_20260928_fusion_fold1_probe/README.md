# BRSET fusion smoke result

> Diagnostic run only: one validation fold, one seed, and a small epoch budget.

- Source revision: `306063eb42c92f92420d404cf44e5be2bbb7dc60`
- Fold: `1`
- Epochs: `3`
- Seed: `42`
- Patient overlap: `0`
- Modalities: `image, table`
- Rules: `1064`
- Intra-modal edges: `1832`
- Cross-modal edges: `2352`
- Majority-class accuracy: `0.9174`

| Method | AUC-ROC | AUC-PR | F1 | BalAcc | Accuracy | ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| FISA sequential | 0.9175 | 0.4969 | 0.3654 | 0.8286 | 0.7273 | 5.6772 |
| FISA lookup | 0.9175 | 0.4969 | 0.3654 | 0.8286 | 0.7273 | 0.0474 |
| FKG-E unsupervised | 0.5369 | 0.1437 | 0.0000 | 0.5000 | 0.9174 | 0.1100 |
| FKG-E full | 0.9491 | 0.6418 | 0.5455 | 0.8869 | 0.8760 | 0.1008 |

These values are pipeline verification evidence, not thesis results. The official protocol still requires five seeds, five folds, nested validation, the root test split, and the remaining objective terms/baselines.
