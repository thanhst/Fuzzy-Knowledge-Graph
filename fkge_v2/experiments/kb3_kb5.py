import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from experiments.common import (FOLD_QUALITY_METRICS, aggregate_runs,
                                benchmark_evaluate, compact_fold_observations,
                                default_fkge_kwargs, fold_bootstrap_summary,
                                prepare_fold, primary_fold_specs)
from models.fisa import FISA
from models.fkge import FKGE


def _validation_contexts():
    specs = primary_fold_specs()
    if C.EVAL.QUICK:
        specs = specs[:1]
    contexts = []
    for spec in specs:
        fkg, train, validation = prepare_fold(spec)
        contexts.append((spec, fkg, train, validation, FISA(fkg, "lookup").fit()))
    return contexts


def run_kb3(dims=None, n_seeds=None):
    dims = list(C.KB3_DIMS if dims is None else dims)
    n_seeds = C.EVAL.N_SEEDS if n_seeds is None else n_seeds
    contexts = _validation_contexts()
    rows = []
    for dimension in dims:
        results = []
        for spec, fkg, train, validation, fisa in contexts:
            for seed_offset in range(n_seeds):
                model = FKGE(fkg, **default_fkge_kwargs(
                    d=dimension, seed=C.FKGE.seed + seed_offset))
                model.fit(fisa_model=fisa, train_samples=train)
                result = benchmark_evaluate(model, validation, reference_model=fisa)
                result["fold"] = spec["fold"]
                result["seed"] = C.FKGE.seed + seed_offset
                results.append(result)
        rows.append({
            "d": dimension,
            **aggregate_runs(results),
            **fold_bootstrap_summary(results, FOLD_QUALITY_METRICS),
            "observations": compact_fold_observations(results, FOLD_QUALITY_METRICS),
            "validation_folds": len(contexts),
            "n_seeds": n_seeds,
        })
        print(f"  d={dimension}: AUC validation={rows[-1]['auc_roc_mean']:.4f}, "
              f"parameters={rows[-1]['n_parameters_mean']:.0f}")

    max_auc = max(row["auc_roc_mean"] for row in rows)
    selected = min(row["d"] for row in rows
                   if row["auc_roc_mean"] >= max_auc - 0.005)
    return {
        "metric": "auc_roc_validation",
        "rows": rows,
        "selected_d": selected,
        "selection_rule": "smallest_d_within_0.005_of_max_validation_auc",
        "test_evaluation_status": "pending_outer_test_frb",
        "ci_method": "fold_bootstrap_percentile_95_seed_mean_within_fold_10000_resamples",
    }


def run_kb5(w_grid=None, k_grid=None, n_seeds=None):
    w_grid = list(C.KB5_W_GRID if w_grid is None else w_grid)
    k_grid = list(C.KB5_K_GRID if k_grid is None else k_grid)
    n_seeds = C.EVAL.N_SEEDS if n_seeds is None else n_seeds
    contexts = _validation_contexts()
    rows = []
    for window in w_grid:
        for negatives in k_grid:
            results = []
            for spec, fkg, train, validation, fisa in contexts:
                for seed_offset in range(n_seeds):
                    model = FKGE(fkg, **default_fkge_kwargs(
                        w=window, K_neg=negatives,
                        seed=C.FKGE.seed + seed_offset))
                    model.fit(fisa_model=fisa, train_samples=train)
                    result = benchmark_evaluate(
                        model, validation, reference_model=fisa)
                    result["fold"] = spec["fold"]
                    result["seed"] = C.FKGE.seed + seed_offset
                    results.append(result)
            row = {
                "w": window,
                "cooccurrence": "full_rule" if window is None else f"window_{window}",
                "K": negatives,
                **aggregate_runs(results),
                **fold_bootstrap_summary(results, FOLD_QUALITY_METRICS),
                "observations": compact_fold_observations(
                    results, FOLD_QUALITY_METRICS),
                "validation_folds": len(contexts),
                "n_seeds": n_seeds,
            }
            rows.append(row)
            print(f"  {row['cooccurrence']}, K={negatives}: AUC validation={row['auc_roc_mean']:.4f}")
    auc_values = [row["auc_roc_mean"] for row in rows]
    return {
        "metric": "auc_roc_validation",
        "rows": rows,
        "auc_range": float(max(auc_values) - min(auc_values)),
        "hypothesis_variation_below_0_02": max(auc_values) - min(auc_values) < 0.02,
        "hypothesis_status": "descriptive_only_requires_sgns_contribution",
        "test_evaluation_status": "pending_outer_test_frb",
        "ci_method": "fold_bootstrap_percentile_95_seed_mean_within_fold_10000_resamples",
    }


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb3_results.json"), "w",
              encoding="utf-8") as stream:
        json.dump(run_kb3(), stream, ensure_ascii=False, indent=2)
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb5_results.json"), "w",
              encoding="utf-8") as stream:
        json.dump(run_kb5(), stream, ensure_ascii=False, indent=2)
