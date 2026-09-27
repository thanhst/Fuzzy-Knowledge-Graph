import argparse
import datetime as dt
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C
from data.pipeline_interface import PrefuzzifiedRulePipeline
from experiments.common import benchmark_evaluate, default_fkge_kwargs, primary_fold_specs
from models.fisa import FISA, majority_class_baseline
from models.fkge import FKGE


METRIC_FIELDS = [
    "accuracy", "balanced_accuracy", "f1", "f1_macro", "auc_roc", "auc_pr",
    "sensitivity", "specificity", "avg_time_per_query_ms", "train_time_s",
    "fit_time_s", "agreement", "mean_kl_divergence", "n_parameters",
]


def _compact(result):
    return {field: result[field] for field in METRIC_FIELDS if field in result}


def _git_revision():
    repository = os.path.dirname(ROOT).replace("\\", "/")
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
            cwd=repository, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_report(payload, path):
    rows = []
    for method, metrics in payload["methods"].items():
        rows.append(
            f"| {method} | {metrics['auc_roc']:.4f} | {metrics['auc_pr']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['accuracy']:.4f} | {metrics['avg_time_per_query_ms']:.4f} |"
        )
    text = "\n".join([
        "# BRSET fusion smoke result",
        "",
        "> Diagnostic run only: one validation fold, one seed, and a small epoch budget.",
        "",
        f"- Source revision: `{payload['source_revision']}`",
        f"- Fold: `{payload['protocol']['fold']}`",
        f"- Epochs: `{payload['protocol']['epochs']}`",
        f"- Seed: `{payload['protocol']['seed']}`",
        f"- Patient overlap: `{payload['data']['patient_overlap_count']}`",
        f"- Modalities: `{', '.join(payload['data']['modalities'])}`",
        f"- Rules: `{payload['graph']['rule_count']}`",
        f"- Intra-modal edges: `{payload['graph']['intra_modal_edge_count']}`",
        f"- Cross-modal edges: `{payload['graph']['cross_modal_edge_count']}`",
        f"- Majority-class accuracy: `{payload['data']['majority_accuracy']:.4f}`",
        "",
        "| Method | AUC-ROC | AUC-PR | F1 | BalAcc | Accuracy | ms/sample |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "These values are pipeline verification evidence, not thesis results. "
        "The official protocol still requires five seeds, five folds, nested validation, "
        "the root test split, and the remaining objective terms/baselines.",
    ])
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "results", "fusion_smoke"),
    )
    args = parser.parse_args()

    specs = primary_fold_specs(modality="fusion")
    spec = next((item for item in specs if item["fold"] == args.fold), None)
    if spec is None:
        raise ValueError(f"Unknown fold {args.fold}; available folds: {[x['fold'] for x in specs]}")

    pipeline = PrefuzzifiedRulePipeline(seed=args.seed)
    fkg, train_samples = pipeline.fit_and_mine(spec["train_records"])
    validation_samples = pipeline.transform(spec["test_records"])

    fisa_sequential = FISA(fkg, inference_mode="sequential").fit()
    fisa_lookup = FISA(fkg, inference_mode="lookup").fit()
    methods = {
        "FISA sequential": _compact(benchmark_evaluate(fisa_sequential, validation_samples)),
        "FISA lookup": _compact(benchmark_evaluate(fisa_lookup, validation_samples)),
    }

    for name, overrides in {
        "FKG-E unsupervised": {"delta_pred": 0.0},
        "FKG-E full": {},
    }.items():
        model = FKGE(
            fkg,
            **default_fkge_kwargs(
                epochs=args.epochs,
                seed=args.seed,
                **overrides,
            ),
        )
        model.fit(fisa_model=fisa_lookup, train_samples=train_samples)
        methods[name] = _compact(
            benchmark_evaluate(model, validation_samples, reference_model=fisa_lookup)
        )

    majority_accuracy, majority_class = majority_class_baseline(validation_samples)
    payload = {
        "status": "smoke_not_official",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": _git_revision(),
        "protocol": {
            "fold": args.fold,
            "epochs": args.epochs,
            "seed": args.seed,
            "timing_repeats": 5,
        },
        "data": {
            "source": spec["metadata"]["source"],
            "modality": spec["metadata"]["modality"],
            "modalities": spec["metadata"]["modalities"],
            "input_graph_kind": spec["metadata"]["input_graph_kind"],
            "train_rows": len(spec["train_records"]),
            "validation_rows": len(spec["test_records"]),
            "patient_overlap_count": spec["metadata"]["patient_overlap_count"],
            "majority_accuracy": majority_accuracy,
            "majority_class": majority_class,
        },
        "graph": {
            "rule_count": len(fkg.rules),
            "token_count": len(fkg.vocab),
            "intra_modal_edge_count": fkg.meta["intra_modal_edge_count"],
            "cross_modal_edge_count": fkg.meta["cross_modal_edge_count"],
        },
        "methods": methods,
        "limitations": [
            "one_validation_fold",
            "one_seed",
            "one_epoch_by_default",
            "not_root_test",
            "missing_extended_objective_terms_and_official_baselines",
        ],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "results.json")
    report_path = os.path.join(args.output_dir, "README.md")
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    _write_report(payload, report_path)
    print(json_path)
    print(report_path)


if __name__ == "__main__":
    main()
