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


SUPPORTED_WEIGHTS = {
    "lambda_S": ("beta_rule", C.FKGE.beta_rule),
    "lambda_N": ("lam_node", C.FKGE.lam_node),
    "lambda_I": ("gamma_inf", C.FKGE.gamma_inf),
    "lambda_P": ("delta_pred", C.FKGE.delta_pred),
    "lambda_C": ("weight_decay", C.FKGE.weight_decay),
}
UNIMPLEMENTED_WEIGHTS = ["lambda_E", "lambda_A", "lambda_B", "lambda_R"]


def _contexts(sample_ratio=1.0):
    specs = primary_fold_specs()
    if C.EVAL.QUICK:
        specs = specs[:1]
    contexts = []
    for spec in specs:
        fkg, train, validation = prepare_fold(
            spec, sample_ratio=sample_ratio, sample_seed=7 + spec["fold"])
        contexts.append((spec, fkg, train, validation, FISA(fkg, "lookup").fit()))
    return contexts


def run_kb4(multipliers=None, n_seeds=None):
    multipliers = list(C.KB4_MULTIPLIERS if multipliers is None else multipliers)
    n_seeds = C.EVAL.N_SEEDS if n_seeds is None else n_seeds
    contexts = _contexts()
    rows = []
    for weight_name, (argument, default_value) in SUPPORTED_WEIGHTS.items():
        for multiplier in multipliers:
            results = []
            for spec, fkg, train, validation, fisa in contexts:
                for seed_offset in range(n_seeds):
                    override = {argument: default_value * multiplier}
                    model = FKGE(fkg, **default_fkge_kwargs(
                        seed=C.FKGE.seed + seed_offset, **override))
                    model.fit(fisa_model=fisa, train_samples=train)
                    result = benchmark_evaluate(
                        model, validation, reference_model=fisa)
                    result["fold"] = spec["fold"]
                    result["seed"] = C.FKGE.seed + seed_offset
                    results.append(result)
            row = {
                "weight": weight_name,
                "argument": argument,
                "default_value": default_value,
                "multiplier": multiplier,
                "effective_value": default_value * multiplier,
                **aggregate_runs(results),
                **fold_bootstrap_summary(results, FOLD_QUALITY_METRICS),
                "observations": compact_fold_observations(
                    results, FOLD_QUALITY_METRICS),
                "validation_folds": len(contexts),
                "n_seeds": n_seeds,
            }
            rows.append(row)
            print(f"  {weight_name} x {multiplier:g}: "
                  f"AUC={row['auc_roc_mean']:.4f}, agreement={row['agreement_mean']:.4f}")
    return {
        "metric": "auc_roc_validation",
        "rows": rows,
        "supported_weights": list(SUPPORTED_WEIGHTS),
        "unimplemented_weights": UNIMPLEMENTED_WEIGHTS,
        "random_search_status": "pending_full_objective_implementation",
        "ci_method": "fold_bootstrap_percentile_95_seed_mean_within_fold_10000_resamples",
    }


def run_kb6(ratios=None, n_seeds=None):
    ratios = list(C.KB6_SAMPLE_RATIOS if ratios is None else ratios)
    n_seeds = C.EVAL.N_SEEDS if n_seeds is None else n_seeds
    specs = primary_fold_specs()
    if C.EVAL.QUICK:
        specs = specs[:1]
    rows = []
    for ratio in ratios:
        sequential_results = []
        lookup_results = []
        fkge_results = []
        rule_counts = []
        for spec in specs:
            fkg, train, validation = prepare_fold(
                spec, sample_ratio=ratio, sample_seed=7 + spec["fold"])
            rule_counts.append(len(fkg))
            sequential = FISA(fkg, "sequential").fit()
            lookup = FISA(fkg, "lookup").fit()
            sequential_result = benchmark_evaluate(sequential, validation)
            lookup_result = benchmark_evaluate(lookup, validation)
            sequential_result["fold"] = spec["fold"]
            lookup_result["fold"] = spec["fold"]
            if sequential_result["y_pred"] != lookup_result["y_pred"]:
                raise AssertionError(
                    f"Fold {spec['fold']}, ratio {ratio}: FISA modes disagree."
                )
            sequential_results.append(sequential_result)
            lookup_results.append(lookup_result)
            for seed_offset in range(n_seeds):
                model = FKGE(fkg, **default_fkge_kwargs(
                    seed=C.FKGE.seed + seed_offset))
                model.fit(fisa_model=lookup, train_samples=train)
                result = benchmark_evaluate(
                    model, validation, reference_model=lookup)
                result["fold"] = spec["fold"]
                result["seed"] = C.FKGE.seed + seed_offset
                fkge_results.append(result)

        row = {
            "ratio": ratio,
            "n_rules_mean": float(np.mean(rule_counts)),
            "n_rules_per_fold": rule_counts,
            **aggregate_runs(sequential_results, prefix="fisa_sequential_"),
            **aggregate_runs(lookup_results, prefix="fisa_lookup_"),
            **aggregate_runs(fkge_results, prefix="fkge_"),
            **fold_bootstrap_summary(
                sequential_results, ("avg_time_per_query_ms", "auc_roc"),
                prefix="fisa_sequential_"),
            **fold_bootstrap_summary(
                lookup_results, ("avg_time_per_query_ms", "auc_roc"),
                prefix="fisa_lookup_"),
            **fold_bootstrap_summary(
                fkge_results, ("avg_time_per_query_ms", "auc_roc"),
                prefix="fkge_"),
            "observations": {
                "fisa_sequential": compact_fold_observations(
                    sequential_results, ("avg_time_per_query_ms", "auc_roc")),
                "fisa_lookup": compact_fold_observations(
                    lookup_results, ("avg_time_per_query_ms", "auc_roc")),
                "fkge": compact_fold_observations(
                    fkge_results, ("avg_time_per_query_ms", "auc_roc")),
            },
            "validation_folds": len(specs),
            "n_seeds": n_seeds,
        }
        rows.append(row)
        print(f"  ratio={ratio:.0%}, |R|={row['n_rules_mean']:.1f}: "
              f"FISA seq={row['fisa_sequential_avg_time_per_query_ms_mean']:.4f} ms, "
              f"lookup={row['fisa_lookup_avg_time_per_query_ms_mean']:.4f} ms, "
              f"FKG-E={row['fkge_avg_time_per_query_ms_mean']:.4f} ms")

    x = np.log([row["n_rules_mean"] for row in rows])
    slopes = {}
    for method, field in {
        "fisa_sequential": "fisa_sequential_avg_time_per_query_ms_mean",
        "fisa_lookup": "fisa_lookup_avg_time_per_query_ms_mean",
        "fkge": "fkge_avg_time_per_query_ms_mean",
    }.items():
        y = np.log([max(row[field], 1e-12) for row in rows])
        slopes[method] = float(np.polyfit(x, y, 1)[0]) if len(rows) > 1 else float("nan")
    return {
        "rows": rows, "log_log_slopes": slopes, "timing_repeats": 5,
        "ci_method": "fold_bootstrap_percentile_95_seed_mean_within_fold_10000_resamples",
    }


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb4_results.json"), "w",
              encoding="utf-8") as stream:
        json.dump(run_kb4(), stream, ensure_ascii=False, indent=2)
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb6_results.json"), "w",
              encoding="utf-8") as stream:
        json.dump(run_kb6(), stream, ensure_ascii=False, indent=2)
