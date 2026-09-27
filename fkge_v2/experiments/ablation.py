import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from experiments.common import (aggregate_runs, benchmark_evaluate,
                                default_fkge_kwargs, prepare_fold,
                                primary_fold_specs)
from models.fisa import FISA
from models.fkge import FKGE


VARIANTS = [
    ("No SGNS", {"beta_rule": 0.0}),
    ("No node loss", {"lam_node": 0.0}),
    ("No FISA distillation", {"gamma_inf": 0.0}),
    ("No label prediction", {"delta_pred": 0.0}),
    ("No L2 regularization", {"weight_decay": 0.0}),
    ("Uniform mean pooling", {"pooling": "mean"}),
    ("Prediction only", {
        "beta_rule": 0.0, "lam_node": 0.0, "gamma_inf": 0.0,
    }),
    ("SGNS only", {
        "lam_node": 0.0, "gamma_inf": 0.0, "delta_pred": 0.0,
    }),
    ("Full FKG-E", {}),
]


def run_ablation(n_seeds=None):
    n_seeds = C.EVAL.N_SEEDS if n_seeds is None else n_seeds
    specs = primary_fold_specs()
    if C.EVAL.QUICK:
        specs = specs[:1]
    contexts = []
    for spec in specs:
        fkg, train, validation = prepare_fold(spec)
        if not fkg.edges:
            raise RuntimeError(f"Fold {spec['fold']} has no edges for node ablation.")
        contexts.append((spec, fkg, train, validation, FISA(fkg, "lookup").fit()))

    rows = []
    for name, overrides in VARIANTS:
        results = []
        for spec, fkg, train, validation, fisa in contexts:
            for seed_offset in range(n_seeds):
                model = FKGE(fkg, **default_fkge_kwargs(
                    seed=C.FKGE.seed + seed_offset, **overrides))
                model.fit(fisa_model=fisa, train_samples=train)
                results.append(benchmark_evaluate(
                    model, validation, reference_model=fisa))
        row = {
            "variant": name,
            "overrides": overrides,
            **aggregate_runs(results),
            "validation_folds": len(contexts),
            "n_seeds": n_seeds,
        }
        rows.append(row)
        print(f"  {name}: AUC={row['auc_roc_mean']:.4f}, "
              f"BalAcc={row['balanced_accuracy_mean']:.4f}, "
              f"agreement={row['agreement_mean']:.4f}")

    full = next(row for row in rows if row["variant"] == "Full FKG-E")
    for row in rows:
        row["delta_auc_vs_full"] = row["auc_roc_mean"] - full["auc_roc_mean"]
        row["delta_agreement_vs_full"] = row["agreement_mean"] - full["agreement_mean"]
    return {
        "rows": rows,
        "metric": "auc_roc_validation",
        "implemented_components": [
            "L_SGNS", "L_node", "L_inf", "L_pred", "L2", "weighted_pooling"
        ],
        "unimplemented_components": ["L_edge", "L_A", "L_B", "L_rule", "attention_pooling"],
    }


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "ablation_results.json"), "w",
              encoding="utf-8") as stream:
        json.dump(run_ablation(), stream, ensure_ascii=False, indent=2)
