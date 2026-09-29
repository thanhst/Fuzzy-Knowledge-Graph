import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from experiments.common import (aggregate_runs, benchmark_evaluate,
                                default_fkge_kwargs, prepare_fold,
                                primary_fold_specs)
from models.fisa import FISA
from models.fkge import FKGE
from models.kge_baselines import (DistMultLite, KNNOnEmbedding, Node2VecLite,
                                  TransELite)
from models.metrics import classification_metrics


def _contexts():
    specs = primary_fold_specs()
    if C.EVAL.QUICK:
        specs = specs[:1]
    contexts = []
    for spec in specs:
        fkg, train, validation = prepare_fold(spec)
        contexts.append((spec, fkg, train, validation, FISA(fkg, "lookup").fit()))
    return contexts


def _fuzzy_matrix(samples, vocab):
    feature_tokens = [token for token in vocab if not token.startswith("class-")]
    matrix = np.asarray([
        [sample["membership"].get(token, 0.0) for token in feature_tokens]
        for sample in samples
    ], dtype=np.float64)
    return matrix, feature_tokens


def _mlp_result(train, validation, fkg, seed):
    from sklearn.neural_network import MLPClassifier

    x_train, feature_tokens = _fuzzy_matrix(train, fkg.vocab)
    x_validation, _ = _fuzzy_matrix(validation, fkg.vocab)
    class_tokens = fkg.class_tokens
    class_to_index = {token: index for index, token in enumerate(class_tokens)}
    y_train = np.asarray([class_to_index[sample["label"]] for sample in train])
    y_validation = np.asarray([class_to_index[sample["label"]] for sample in validation])
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                          early_stopping=True, random_state=seed)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - started
    started = time.perf_counter()
    probability = model.predict_proba(x_validation)
    inference_time = time.perf_counter() - started
    predicted = np.argmax(probability, axis=1)
    y_true = [class_tokens[index] for index in y_validation]
    y_pred = [class_tokens[index] for index in predicted]
    positive_score = probability[:, class_to_index[class_tokens[-1]]].tolist()
    result = classification_metrics(y_true, y_pred, positive_score, class_tokens)
    result.update({
        "train_time_s": train_time,
        "total_time_s": inference_time,
        "avg_time_per_query_ms": inference_time / len(validation) * 1000,
        "n_parameters": int(sum(weight.size for weight in model.coefs_)
                            + sum(bias.size for bias in model.intercepts_)),
        "feature_count": len(feature_tokens),
    })
    return result


def run_baseline_comparison(n_seeds=None):
    n_seeds = C.EVAL.N_SEEDS if n_seeds is None else n_seeds
    contexts = _contexts()
    collected = {
        "FISA lookup": [],
        "DeepWalk-lite + kNN": [],
        "TransE-lite + kNN": [],
        "DistMult-lite + kNN": [],
        "MLP fuzzy features": [],
        "FKG-E unsupervised": [],
        "FKG-E full": [],
    }
    fold_results = {method: {} for method in collected}

    def record(method, fold, result):
        collected[method].append(result)
        fold_results[method].setdefault(fold, []).append(result)

    for spec, fkg, train, validation, fisa in contexts:
        fisa_result = benchmark_evaluate(fisa, validation)
        fisa_result["train_time_s"] = fisa_result["fit_time_s"]
        record("FISA lookup", spec["fold"], fisa_result)

        for seed_offset in range(n_seeds):
            seed = C.FKGE.seed + seed_offset
            embeddings = [
                ("DeepWalk-lite + kNN", Node2VecLite(
                    fkg, d=C.FKGE.d, epochs=15, seed=seed)),
                ("TransE-lite + kNN", TransELite(
                    fkg, d=C.FKGE.d, epochs=15, seed=seed)),
                ("DistMult-lite + kNN", DistMultLite(
                    fkg, d=C.FKGE.d, epochs=15, seed=seed)),
            ]
            for method, embedding in embeddings:
                embedding.fit()
                classifier = KNNOnEmbedding(
                    fkg, embedding.E, fkg.token2idx, k=5)
                result = benchmark_evaluate(classifier, validation)
                result["train_time_s"] = embedding.train_time_s
                result["n_parameters"] = int(embedding.E.size)
                record(method, spec["fold"], result)

            record("MLP fuzzy features", spec["fold"],
                   _mlp_result(train, validation, fkg, seed))
            for method, overrides in {
                "FKG-E unsupervised": {"delta_pred": 0.0},
                "FKG-E full": {},
            }.items():
                model = FKGE(fkg, **default_fkge_kwargs(seed=seed, **overrides))
                model.fit(fisa_model=fisa, train_samples=train)
                record(method, spec["fold"], benchmark_evaluate(
                    model, validation, reference_model=fisa))

    rows = []
    for method, results in collected.items():
        row = {
            "method": method,
            "status": "completed",
            "official_baseline": method in {
                "FISA lookup", "MLP fuzzy features", "FKG-E unsupervised", "FKG-E full"
            },
            **aggregate_runs(results),
        }
        for metric in ("auc_roc", "balanced_accuracy", "f1", "agreement",
                       "fidelity_balanced_accuracy", "cohen_kappa"):
            means = [float(np.mean([result[metric] for result in per_seed
                                    if metric in result and np.isfinite(result[metric])]))
                     for per_seed in fold_results[method].values()
                     if any(metric in result and np.isfinite(result[metric])
                            for result in per_seed)]
            if means:
                sampled = np.random.RandomState(42).choice(
                    means, size=(10000, len(means)), replace=True)
                low, high = np.percentile(sampled.mean(axis=1), [2.5, 97.5])
                row[f"{metric}_ci95_low"] = float(low)
                row[f"{metric}_ci95_high"] = float(high)
        rows.append(row)

    rows.extend([
        {
            "method": "Node2Vec + kNN",
            "status": "not_implemented_node2vec_bias_parameters",
            "official_baseline": False,
        },
        {
            "method": "XGBoost fuzzy/original features",
            "status": ("dependency_available_not_integrated" if importlib.util.find_spec("xgboost")
                       else "unavailable_dependency_xgboost"),
            "official_baseline": False,
        },
        {
            "method": "TransE/DistMult (PyKEEN standard)",
            "status": ("dependency_available_not_run" if importlib.util.find_spec("pykeen")
                       else "unavailable_dependency_pykeen"),
            "official_baseline": False,
        },
    ])
    return {
        "rows": rows,
        "validation_folds": len(contexts),
        "n_seeds": n_seeds,
        "lite_models_are_smoke_only": True,
        "ci_method": "fold_bootstrap_percentile_95_seed_mean_within_fold_10000_resamples",
    }


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "baseline_comparison.json"), "w",
              encoding="utf-8") as stream:
        json.dump(run_baseline_comparison(), stream, ensure_ascii=False, indent=2)
