import os
import statistics

import numpy as np

import config as C
from data.fkg_io import load_raw_records, generate_synthetic_prefuzzified_records
from data.frb_package import load_frb_folds, package_available
from data.kfold_utils import make_patient_kfold
from data.pipeline_interface import PrefuzzifiedRulePipeline


FOLD_QUALITY_METRICS = (
    "auc_roc", "auc_pr", "f1", "balanced_accuracy", "agreement",
    "fidelity_balanced_accuracy", "cohen_kappa", "mean_kl_divergence",
)


def primary_fold_specs(modality=None):
    modality = modality or C.BRSET_PRIMARY_MODALITY
    if package_available(C.PATHS.BRSET_FRB_PACKAGE):
        folds = load_frb_folds(C.PATHS.BRSET_FRB_PACKAGE, modality=modality)
        if modality == "fusion":
            for fold in folds:
                if fold["metadata"]["input_graph_kind"] != "FKG-MM":
                    raise ValueError(
                        f"Fold {fold['fold']} is not multimodal: "
                        f"{fold['metadata']['modalities']}"
                    )
        return folds

    if os.path.exists(C.PATHS.BRSET_RAW_FILE):
        records = load_raw_records(C.PATHS.BRSET_RAW_FILE)
        source = "brset_raw_json"
        synthetic = False
    else:
        records = generate_synthetic_prefuzzified_records(n_patients=300, seed=100)
        source = "synthetic_prefuzzified"
        synthetic = True

    specs = []
    for fold_index, (train_idx, test_idx) in enumerate(
            make_patient_kfold(records, k=C.EVAL.K_FOLD, seed=42), start=1):
        train_records = [records[index] for index in train_idx]
        test_records = [records[index] for index in test_idx]
        train_patients = {record["patient_id"] for record in train_records}
        test_patients = {record["patient_id"] for record in test_records}
        overlap = train_patients & test_patients
        if overlap:
            raise ValueError(f"Fold {fold_index} has {len(overlap)} overlapping patients.")
        specs.append({
            "fold": fold_index,
            "train_records": train_records,
            "test_records": test_records,
            "metadata": {
                "source": source,
                "is_synthetic": synthetic,
                "train_patient_count": len(train_patients),
                "test_patient_count": len(test_patients),
                "patient_overlap_count": 0,
            },
        })
    return specs


def prepare_fold(spec, sample_ratio=1.0, sample_seed=0):
    pipeline = PrefuzzifiedRulePipeline(sample_ratio=sample_ratio, seed=sample_seed)
    fkg, train_samples = pipeline.fit_and_mine(spec["train_records"])
    test_samples = pipeline.transform(spec["test_records"])
    return fkg, train_samples, test_samples


def default_fkge_kwargs(**overrides):
    values = {
        "d": C.FKGE.d,
        "w": C.FKGE.w,
        "K_neg": C.FKGE.K_neg,
        "lam_node": C.FKGE.lam_node,
        "beta_rule": C.FKGE.beta_rule,
        "gamma_inf": C.FKGE.gamma_inf,
        "delta_pred": C.FKGE.delta_pred,
        "weight_decay": C.FKGE.weight_decay,
        "aggregation": C.FKGE.aggregation,
        "max_pairs_per_epoch": C.FKGE.max_pairs_per_epoch,
        "lr": C.FKGE.lr,
        "epochs": C.FKGE.epochs,
        "pooling": C.FKGE.pooling,
        "alpha_pool": C.FKGE.alpha_pool,
        "seed": C.FKGE.seed,
    }
    values.update(overrides)
    return values


def benchmark_evaluate(model, samples, reference_model=None, repeats=5):
    model.evaluate(samples, reference_model=reference_model) if reference_model is not None else model.evaluate(samples)
    results = [
        model.evaluate(samples, reference_model=reference_model)
        if reference_model is not None else model.evaluate(samples)
        for _ in range(repeats)
    ]
    result = dict(results[0])
    result["avg_time_per_query_ms"] = float(statistics.median(
        item["avg_time_per_query_ms"] for item in results))
    result["total_time_s"] = float(statistics.median(
        item["total_time_s"] for item in results))
    result["timing_repeats"] = repeats
    return result


def aggregate_runs(results, prefix=""):
    fields = [
        "accuracy", "balanced_accuracy", "f1", "f1_macro", "auc_roc", "auc_pr",
        "sensitivity", "specificity", "avg_time_per_query_ms", "total_time_s",
        "train_time_s", "representation_build_time_s", "agreement",
        "fidelity_balanced_accuracy", "cohen_kappa", "fidelity_bound_coverage",
        "fidelity_bound_violations",
        "mean_kl_divergence", "n_parameters", "embedding_memory_bytes",
    ]
    output = {}
    for field in fields:
        values = [float(result[field]) for result in results
                  if field in result and result[field] is not None
                  and not np.isnan(float(result[field]))]
        if not values:
            continue
        key = f"{prefix}{field}" if prefix else field
        output[f"{key}_mean"] = float(np.mean(values))
        output[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return output


def fold_bootstrap_summary(results, metrics, prefix=""):
    """Bootstrap patient folds after averaging seeds within each fold."""
    output = {}
    for metric in metrics:
        folds = sorted({int(result["fold"]) for result in results})
        fold_means = []
        for fold in folds:
            values = [float(result[metric]) for result in results
                      if int(result["fold"]) == fold and metric in result
                      and result[metric] is not None
                      and np.isfinite(float(result[metric]))]
            if values:
                fold_means.append(float(np.mean(values)))
        if not fold_means:
            continue
        key = f"{prefix}{metric}"
        output[f"{key}_fold_std"] = (
            float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0)
        draws = np.random.RandomState(42).choice(
            fold_means, size=(10000, len(fold_means)), replace=True).mean(axis=1)
        low, high = np.percentile(draws, [2.5, 97.5])
        output[f"{key}_ci95_low"] = float(low)
        output[f"{key}_ci95_high"] = float(high)
    return output


def compact_fold_observations(results, metrics):
    """Persist the small fold/seed metric records needed to audit intervals."""
    observations = []
    for result in results:
        row = {"fold": int(result["fold"])}
        if "seed" in result:
            row["seed"] = int(result["seed"])
        for metric in metrics:
            if metric in result and result[metric] is not None:
                row[metric] = float(result[metric])
        observations.append(row)
    return observations
