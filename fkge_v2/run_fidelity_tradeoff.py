"""Diagnostic 5-fold sweep of lambda_I/lambda_P on the available objective.

The default gamma=2 run can be reused from a matching KB1 result. This is
validation on the root-train folds, not nested outer-test evaluation.
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from experiments.common import benchmark_evaluate, default_fkge_kwargs, prepare_fold, primary_fold_specs
from models.fisa import FISA
from models.fkge import FKGE
from report.run_manifest import create_manifest, finish_manifest, write_manifest


GAMMA_VALUES = (0.0, 2.0, 20.0, 100.0)
METRICS = ("auc_roc", "balanced_accuracy", "f1", "agreement",
           "fidelity_balanced_accuracy", "cohen_kappa", "mean_kl_divergence",
           "fidelity_bound_coverage")


def _load_matching_kb1(run_id):
    if os.path.basename(run_id) != run_id:
        raise ValueError("kb1 run ID must be a single directory name")
    root = os.path.join(C.PATHS.ROOT, "outputs", "runs", run_id)
    with open(os.path.join(root, "run_manifest.json"), encoding="utf-8") as stream:
        manifest = json.load(stream)
    with open(os.path.join(root, "kb1_results.json"), encoding="utf-8") as stream:
        datasets = json.load(stream)
    if len(datasets) != 1 or datasets[0]["dataset"] != "BRSET fusion":
        raise ValueError("Expected exactly one BRSET fusion KB1 dataset")
    config = manifest["configuration"]
    expected = {"epochs": C.FKGE.epochs, "n_seeds": C.EVAL.N_SEEDS,
                "k_fold": C.EVAL.K_FOLD, "lambda_inf": C.FKGE.gamma_inf,
                "lambda_pred": C.FKGE.delta_pred, "aggregation": C.FKGE.aggregation,
                "max_pairs_per_epoch": C.FKGE.max_pairs_per_epoch}
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"KB1 configuration {field} differs: {config.get(field)} != {value}")
    reused = [row for row in datasets[0]["observations"]
              if row["method"] == "FKG-E full"]
    if len(reused) != C.EVAL.K_FOLD * C.EVAL.N_SEEDS:
        raise ValueError("KB1 does not contain all fold x seed observations")
    return {(row["fold"], row["seed"]): row for row in reused}


def _summarize(gamma, observations):
    row = {"lambda_I": gamma, "lambda_P": C.FKGE.delta_pred,
           "lambda_I_over_lambda_P": gamma / C.FKGE.delta_pred,
           "n_folds": C.EVAL.K_FOLD, "n_seeds": C.EVAL.N_SEEDS}
    for metric in METRICS:
        fold_means = [np.mean([item[metric] for item in observations
                               if item["fold"] == fold])
                      for fold in range(1, C.EVAL.K_FOLD + 1)]
        if not np.all(np.isfinite(fold_means)):
            raise ValueError(f"Non-finite {metric} at lambda_I={gamma}")
        row[f"{metric}_mean"] = float(np.mean(fold_means))
        row[f"{metric}_std"] = float(np.std(
            [item[metric] for item in observations], ddof=1))
        draws = np.random.RandomState(42).choice(
            fold_means, size=(10000, len(fold_means)), replace=True).mean(axis=1)
        low, high = np.percentile(draws, [2.5, 97.5])
        row[f"{metric}_ci95_low"] = float(low)
        row[f"{metric}_ci95_high"] = float(high)
    return row


def _write_outputs(output_dir, rows, observations, reused_run):
    payload = {"objective_status": "partial_L_SGNS_L_node_L_inf_L_pred_L2_only",
               "teacher_calibration": "not_available_for_patient_linked_inner_validation",
               "decision_threshold": "argmax_uncalibrated",
               "evaluation": "five_root_train_validation_folds_not_outer_test",
               "ci_method": "fold_bootstrap_percentile_95_seed_mean_within_fold_10000_resamples",
               "reused_kb1_run": reused_run, "rows": rows, "observations": observations}
    with open(os.path.join(output_dir, "tradeoff_results.json"), "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    fields = ["lambda_I_over_lambda_P", "lambda_I", "lambda_P"] + [
        f"{metric}_{suffix}" for metric in METRICS
        for suffix in ("mean", "std", "ci95_low", "ci95_high")]
    with open(os.path.join(output_dir, "table_tradeoff.csv"), "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    for row in rows:
        x, y = row["fidelity_balanced_accuracy_mean"], row["auc_roc_mean"]
        axis.errorbar(x, y,
                      xerr=[[x - row["fidelity_balanced_accuracy_ci95_low"]],
                            [row["fidelity_balanced_accuracy_ci95_high"] - x]],
                      yerr=[[y - row["auc_roc_ci95_low"]],
                            [row["auc_roc_ci95_high"] - y]],
                      fmt="o", capsize=3,
                      label=f"λI/λP={row['lambda_I_over_lambda_P']:g}")
    axis.set_xlabel("Trung thành cân bằng với FISA")
    axis.set_ylabel("AUC-ROC")
    axis.set_title("Đánh đổi trên objective hiện có; validation 5-fold")
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tradeoff_auc_fidelity.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb1-run-id", required=True)
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1 or os.path.basename(args.run_id) != args.run_id:
        parser.error("epochs must be positive and run ID must be one directory name")
    C.FKGE.epochs = args.epochs
    reused = _load_matching_kb1(args.kb1_run_id)
    output_dir = os.path.join(C.PATHS.ROOT, "outputs", "runs", args.run_id)
    C.PATHS.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest = create_manifest(args.run_id, False, ["fidelity_tradeoff"], sys.argv)
    manifest["configuration"]["lambda_I_grid"] = GAMMA_VALUES
    manifest["reused_kb1_run"] = args.kb1_run_id
    write_manifest(manifest)
    started = time.time()
    observations = []
    rows = []
    try:
        for gamma in GAMMA_VALUES:
            print(f"λI={gamma:g}, λI/λP={gamma / C.FKGE.delta_pred:g}", flush=True)
            for spec in primary_fold_specs():
                if gamma == C.FKGE.gamma_inf:
                    for seed_offset in range(C.EVAL.N_SEEDS):
                        seed = C.FKGE.seed + seed_offset
                        observations.append(dict(reused[(spec["fold"], seed)],
                                                 lambda_I=gamma, source="reused_kb1"))
                    continue
                fkg, train, validation = prepare_fold(spec, sample_seed=spec["fold"])
                teacher = FISA(fkg, "lookup").fit()
                for seed_offset in range(C.EVAL.N_SEEDS):
                    seed = C.FKGE.seed + seed_offset
                    model = FKGE(fkg, **default_fkge_kwargs(
                        gamma_inf=gamma, seed=seed))
                    model.fit(fisa_model=teacher, train_samples=train)
                    result = benchmark_evaluate(model, validation, reference_model=teacher)
                    observations.append({"fold": spec["fold"], "seed": seed,
                                         "lambda_I": gamma, "source": "new_fit",
                                         **{metric: result[metric] for metric in METRICS}})
                print(f"  fold={spec['fold']} completed", flush=True)
            selected = [item for item in observations if item["lambda_I"] == gamma]
            rows.append(_summarize(gamma, selected))
            _write_outputs(output_dir, rows, observations, args.kb1_run_id)
            print(f"  AUC={rows[-1]['auc_roc_mean']:.4f}, "
                  f"fidelity_bal={rows[-1]['fidelity_balanced_accuracy_mean']:.4f}",
                  flush=True)
        finish_manifest(manifest, time.time() - started)
    except Exception as exc:
        finish_manifest(manifest, time.time() - started, status="failed", error=exc)
        raise
    finally:
        write_manifest(manifest)


if __name__ == "__main__":
    main()
