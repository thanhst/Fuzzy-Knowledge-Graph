import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from data.fkg_io import generate_synthetic_raw_records, load_raw_records
from data.kfold_utils import make_patient_kfold
from data.pipeline_interface import RealisticSyntheticPipeline, PrefuzzifiedRulePipeline
from experiments.common import (aggregate_runs, benchmark_evaluate,
                                default_fkge_kwargs, primary_fold_specs)
from models.fisa import FISA, warn_if_not_beating_majority
from models.fkge import FKGE


def _raw_dataset_specs(raw_path, dataset_name, seed):
    if os.path.exists(raw_path):
        records = load_raw_records(raw_path)
        source = os.path.abspath(raw_path)
        synthetic = False
    else:
        records = generate_synthetic_raw_records(
            n_patients=200, seed=seed, dataset_name=dataset_name)
        source = "synthetic_raw"
        synthetic = True

    specs = []
    for fold, (train_idx, test_idx) in enumerate(
            make_patient_kfold(records, k=C.EVAL.K_FOLD, seed=42), start=1):
        specs.append({
            "fold": fold,
            "train_records": [records[index] for index in train_idx],
            "test_records": [records[index] for index in test_idx],
            "metadata": {
                "source": source,
                "is_synthetic": synthetic,
                "patient_overlap_count": 0,
            },
        })
    return specs


def _run_fold_specs(specs, dataset_name, tag, pipeline_factory,
                    n_seeds_per_fold=None, sample_ratio=1.0):
    n_seeds_per_fold = n_seeds_per_fold or C.EVAL.N_SEEDS
    method_results = {
        "FISA sequential": [],
        "FISA lookup": [],
        "FKG-E unsupervised": [],
        "FKG-E full": [],
    }
    fold_results = {method: {} for method in method_results}
    observations = []
    rule_counts = []
    rule_class_counts = []

    for spec in specs:
        pipeline = pipeline_factory(sample_ratio, spec["fold"])
        fkg, train_samples = pipeline.fit_and_mine(spec["train_records"])
        test_samples = pipeline.transform(spec["test_records"])
        if not fkg.edges:
            raise RuntimeError(f"Fold {spec['fold']} produced no FKG edges; L_node is undefined.")
        rule_counts.append(len(fkg))
        rule_class_counts.append({
            label: sum(rule["consequent_token"] == label for rule in fkg.rules)
            for label in fkg.class_tokens
        })

        fisa_sequential = FISA(fkg, inference_mode="sequential").fit()
        fisa_lookup = FISA(fkg, inference_mode="lookup").fit()
        sequential_result = benchmark_evaluate(fisa_sequential, test_samples)
        lookup_result = benchmark_evaluate(fisa_lookup, test_samples)
        method_results["FISA sequential"].append(sequential_result)
        method_results["FISA lookup"].append(lookup_result)
        fold_results["FISA sequential"][spec["fold"]] = [sequential_result]
        fold_results["FISA lookup"][spec["fold"]] = [lookup_result]

        if sequential_result["y_pred"] != lookup_result["y_pred"]:
            raise AssertionError(
                f"Fold {spec['fold']}: sequential and lookup FISA predictions differ."
            )

        for seed_offset in range(n_seeds_per_fold):
            seed = C.FKGE.seed + seed_offset
            variants = {
                "FKG-E unsupervised": {"delta_pred": 0.0},
                "FKG-E full": {},
            }
            for method, overrides in variants.items():
                model = FKGE(fkg, **default_fkge_kwargs(seed=seed, **overrides))
                model.fit(fisa_model=fisa_lookup, train_samples=train_samples)
                result = benchmark_evaluate(
                    model, test_samples, reference_model=fisa_lookup)
                warn_if_not_beating_majority(
                    result, test_samples,
                    model_name=f"{method} ({dataset_name}, fold={spec['fold']}, seed={seed})")
                method_results[method].append(result)
                fold_results[method].setdefault(spec["fold"], []).append(result)
                observations.append({
                    "fold": spec["fold"],
                    "seed": seed,
                    "method": method,
                    "auc_roc": result["auc_roc"],
                    "f1": result["f1"],
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "agreement": result["agreement"],
                    "fidelity_balanced_accuracy": result["fidelity_balanced_accuracy"],
                    "cohen_kappa": result["cohen_kappa"],
                    "fidelity_bound_coverage": result["fidelity_bound_coverage"],
                    "mean_kl_divergence": result["mean_kl_divergence"],
                    "avg_time_per_query_ms": result["avg_time_per_query_ms"],
                })

    methods = []
    for method, results in method_results.items():
        row = {"method": method, **aggregate_runs(results)}
        # The fold, rather than the seed, is the independent unit for the
        # interval. First average seeds within each fold, then bootstrap folds.
        for metric in ("auc_roc", "balanced_accuracy", "agreement",
                       "fidelity_balanced_accuracy", "cohen_kappa"):
            fold_means = [float(np.mean([result[metric] for result in per_seed
                                         if metric in result and np.isfinite(result[metric])]))
                          for per_seed in fold_results[method].values()
                          if any(metric in result and np.isfinite(result[metric])
                                 for result in per_seed)]
            if fold_means:
                rng = np.random.RandomState(42)
                sampled = rng.choice(fold_means, size=(10000, len(fold_means)), replace=True)
                low, high = np.percentile(sampled.mean(axis=1), [2.5, 97.5])
                row[f"{metric}_ci95_low"] = float(low)
                row[f"{metric}_ci95_high"] = float(high)
        if method.startswith("FISA"):
            row["train_time_s_mean"] = float(np.mean(
                [result["fit_time_s"] for result in results]))
            row["train_time_s_std"] = float(np.std(
                [result["fit_time_s"] for result in results], ddof=1)) if len(results) > 1 else 0.0
        methods.append(row)

    source_metadata = [dict(spec["metadata"], fold=spec["fold"]) for spec in specs]
    synthetic = any(item.get("is_synthetic", False) for item in source_metadata)
    return {
        "dataset": dataset_name,
        "kb": tag,
        "source": source_metadata[0].get("source"),
        "is_synthetic": synthetic,
        "k_fold": len(specs),
        "n_seeds_per_fold": n_seeds_per_fold,
        "n_rules_mean": float(np.mean(rule_counts)),
        "n_rules_per_fold": rule_counts,
        "n_rules_by_class_per_fold": rule_class_counts,
        "patient_overlap_count": max(
            item.get("patient_overlap_count", 0) for item in source_metadata),
        "objective_status": "partial_L_SGNS_L_node_L_inf_L_pred_L2_only",
        "teacher_calibration": "not_available_without_patient_linked_inner_training_rows",
        "decision_threshold": "argmax_uncalibrated",
        "ci_method": "fold_bootstrap_percentile_95_seed_mean_within_fold_10000_resamples",
        "methods": methods,
        "observations": observations,
        "fold_metadata": source_metadata,
    }


def run_kb1(brset_only=False):
    print("\n" + "=" * 72)
    print("KB1: FKG-E vs FISA sequential/lookup on the same fold-specific FKG")
    print("=" * 72)
    datasets = []
    if not brset_only:
        datasets.extend([
            (
                "Diabetes-Kaggle",
                _raw_dataset_specs(C.PATHS.DIABETES_KAGGLE_RAW_FILE,
                                   "Diabetes-Kaggle", 101),
                lambda ratio, seed: RealisticSyntheticPipeline(
                    seed=seed, sample_ratio=ratio),
            ),
            (
                "Healthcare-Diabetes-Kaggle",
                _raw_dataset_specs(C.PATHS.HEALTHCARE_DIABETES_RAW_FILE,
                                   "Healthcare-Diabetes-Kaggle", 102),
                lambda ratio, seed: RealisticSyntheticPipeline(
                    seed=seed, sample_ratio=ratio),
            ),
        ])
    datasets.append((
        f"BRSET {C.BRSET_PRIMARY_MODALITY}",
        primary_fold_specs(),
        lambda ratio, seed: PrefuzzifiedRulePipeline(
            seed=seed, sample_ratio=ratio),
    ))
    return [
        _run_fold_specs(specs, name, "KB1", pipeline_factory)
        for name, specs, pipeline_factory in datasets
    ]


def run_kb2():
    print("\n" + "=" * 72)
    print(f"KB2: FKG/FKGS x FISA/FKG-E on BRSET {C.BRSET_PRIMARY_MODALITY} folds")
    print("=" * 72)
    specs = primary_fold_specs()
    pipeline_factory = lambda ratio, seed: PrefuzzifiedRulePipeline(
        seed=seed, sample_ratio=ratio,
        min_class_fraction=0.4 if ratio < 1 else 0.0)
    full = _run_fold_specs(specs, f"BRSET {C.BRSET_PRIMARY_MODALITY} / FKG", "KB2-full",
                           pipeline_factory, sample_ratio=1.0)
    sampled = _run_fold_specs(specs, f"BRSET {C.BRSET_PRIMARY_MODALITY} / FKGS 30%", "KB2-sampled",
                              pipeline_factory, sample_ratio=0.3)

    def method_value(result, method, field):
        row = next(item for item in result["methods"] if item["method"] == method)
        return row[field]

    full_auc = method_value(full, "FKG-E full", "auc_roc_mean")
    sampled_auc = method_value(sampled, "FKG-E full", "auc_roc_mean")
    full_balacc = method_value(full, "FKG-E full", "balanced_accuracy_mean")
    sampled_balacc = method_value(sampled, "FKG-E full", "balanced_accuracy_mean")
    sampled_teacher_balacc = method_value(sampled, "FISA lookup", "balanced_accuracy_mean")
    sampled_agreement = method_value(sampled, "FKG-E full", "agreement_mean")
    full_time = method_value(full, "FKG-E full", "avg_time_per_query_ms_mean")
    sampled_time = method_value(sampled, "FKG-E full", "avg_time_per_query_ms_mean")
    verdict = {
        "sampled_min_class_fraction": 0.4,
        "delta_auc_sampled_minus_full": sampled_auc - full_auc,
        "delta_balanced_accuracy_sampled_minus_full": sampled_balacc - full_balacc,
        "sampled_fisa_balanced_accuracy": sampled_teacher_balacc,
        "sampled_fkge_fisa_agreement": sampled_agreement,
        "inference_speed_ratio_full_over_sampled": full_time / max(sampled_time, 1e-12),
        "interpretation": (
            "fisa_teacher_collapsed_fidelity_unresolved" if sampled_teacher_balacc <= 0.52
            else "quality_cost_descriptive_only"
            if sampled_auc >= full_auc - 0.02
            and sampled_balacc >= full_balacc - 0.02 and sampled_time < full_time
            else "triet_tieu_hoac_khong_ro"
        ),
    }
    return [full, sampled], verdict


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb1_results.json"), "w",
              encoding="utf-8") as stream:
        json.dump(run_kb1(), stream, ensure_ascii=False, indent=2)
    rows, verdict = run_kb2()
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb2_results.json"), "w",
              encoding="utf-8") as stream:
        json.dump({"rows": rows, "verdict": verdict}, stream,
                  ensure_ascii=False, indent=2)
