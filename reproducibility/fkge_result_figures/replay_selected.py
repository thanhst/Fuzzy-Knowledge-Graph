"""Replay five selected configurations from the new private FKG-E package.

This reproduces the supplied protocol, including its documented inner-validation
limitations. It does not repeat hyperparameter selection or certify the protocol.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=lambda item: item.tolist()) + "\n", encoding="utf-8")
    temporary.replace(path)


def replay_fold(package_path, rules_path, output_path, fold, seed, epochs):
    started = time.time()
    package = Path(package_path)
    output = Path(output_path)
    sys.path.insert(0, str(package / "fkge_branch"))
    import numpy as np
    from experiments import nested as nested

    expected = json.loads((package / "expected/fkge_tuned" / fold / "tuned_results.json").read_text(encoding="utf-8"))
    selected = expected["selected"]
    overrides = {key: selected[key] for key in expected["grid"]}
    print(f"START {fold}: {overrides}", flush=True)
    nf = nested.NestedFold(rules_path, fold, seed=seed, split_unit="patient")
    teacher = nested.cached_teacher(nf, "full", str(output / "cache"))
    model, metrics = nf.fit_fkge(teacher, seed=seed, T_ep=epochs, **nested.canon(overrides))
    fisa_metrics = nf.fisa_metrics(teacher)
    q = model._query_matrix(nf.test)
    _, probabilities = model._infer(q)
    score_e = model._logratio(probabilities)
    score_f = []
    for sample in nf.test:
        values, _, matched = teacher.fisa.decision_values(sample["membership"])
        score_f.append(teacher.fisa._score(values) if matched else 0.0)
    truth = np.array([int(sample["label"] == nf.pos) for sample in nf.test])
    with (Path(rules_path) / fold / "test_ids.csv").open(encoding="utf-8-sig", newline="") as stream:
        ids = list(csv.DictReader(stream))
    if len(ids) != len(truth):
        raise AssertionError("Prediction and image ID counts differ.")
    directory = output / fold
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "predictions_PRIVATE.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "patient_id", "y_true", "score_fisa", "score_fkge"])
        writer.writerows((row["image_id"], row["patient_id"], int(y), float(sf), float(se))
                         for row, y, sf, se in zip(ids, truth, score_f, score_e))
    differences = {
        "fkge_roc_auc": float(metrics["auc"] - selected["auc_mean"]),
        "fkge_ap": float(metrics["pr_auc"] - selected["pr_auc_mean"]),
        "fisa_roc_auc": float(fisa_metrics["auc"] - expected["fisa"]["auc"]),
        "fisa_ap": float(fisa_metrics["pr_auc"] - expected["fisa"]["pr_auc"]),
    }
    result = {
        "fold": fold,
        "selected_parameters": overrides,
        "seed": seed,
        "maximum_epochs": epochs,
        "info": nf.info,
        "fkge": metrics,
        "fisa": fisa_metrics,
        "expected_selected_metrics": {key: selected[key] for key in ("auc_mean", "pr_auc_mean", "val_pred_loss_mean")},
        "differences_from_expected": differences,
        "matches_expected_within_1e_6": all(abs(value) <= 1e-6 for value in differences.values()),
        "history": {key: list(map(float, values)) for key, values in model.history.items()},
        "seconds": time.time() - started,
    }
    write_json(directory / "replay_result.json", result)
    print(f"DONE {fold}: ROC={metrics['auc']:.6f}, AP={metrics['pr_auc']:.6f}, "
          f"delta_AP={differences['fkge_ap']:+.3g}, {result['seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private directory; contains patient scores.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    package = args.package_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new/empty output directory so no stale cache or result is reused.")
    if args.workers < 1:
        parser.error("--workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = package / "expected/fkge_tuned/run_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = original["config"]
    if len(cfg["seeds"]) != 1:
        parser.error("This replay supports the supplied one-seed selection only.")
    sys.path.insert(0, str(package))
    from scripts.export_fkge_folds import export

    rules = output / "generated_folds_PRIVATE"
    export(str(package / "data/KFold_feature_selection_rerun_20260921"), str(rules))
    expected_hashes = package / "environment/folds_sha256.txt"
    hash_checks = []
    for line in expected_hashes.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected_hash, name = line.split(None, 1)
        relative = Path(name.strip()).relative_to("fkge_folds")
        data = (rules / relative).read_bytes()
        raw_hash = hashlib.sha256(data).hexdigest()
        lf_hash = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        hash_checks.append({"file": relative.as_posix(), "raw_match": raw_hash == expected_hash,
                            "lf_normalized_match": lf_hash == expected_hash})
    if not all(row["raw_match"] or row["lf_normalized_match"] for row in hash_checks):
        raise AssertionError("Exported folds differ from the expected folds beyond CSV line endings.")
    package_names = ["numpy", "pandas", "scipy", "scikit-learn", "autograd", "matplotlib"]
    manifest = {
        "status": "running",
        "protocol": "Replay of selected configurations from the supplied tuned zip; protocol not corrected.",
        "selection_rerun": False,
        "source_manifest_sha256": digest(manifest_path),
        "source_code_sha256": original["code_sha256"],
        "folds": original["folds"],
        "seed": cfg["seeds"][0],
        "maximum_epochs": cfg["T_ep"],
        "workers": args.workers,
        "fold_hash_checks": hash_checks,
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                        "blas_threads_per_worker": 1,
                        "packages": {name: importlib.metadata.version(name) for name in package_names}},
        "limitations": [
            "1,208-image subset; not a complete-BRSET model experiment.",
            "Inner validation follows the supplied preprocessed/SMOTE data and full rule graph.",
            "Five configurations already selected in the package; 108-configuration search not repeated.",
            "Windows/Python may differ from the author's original Linux/Python 3.12 runtime.",
        ],
    }
    write_json(output / "replay_manifest.json", manifest)
    started = time.time()
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            tasks = [pool.submit(replay_fold, str(package), str(rules), str(output), fold,
                                 cfg["seeds"][0], cfg["T_ep"]) for fold in original["folds"]]
            for task in as_completed(tasks):
                results.append(task.result())
    except BaseException as error:
        manifest.update(status="failed", error=str(error), seconds=time.time() - started)
        write_json(output / "replay_manifest.json", manifest)
        raise
    results.sort(key=lambda row: row["fold"])
    manifest.update(status="completed", seconds=time.time() - started,
                    matches_expected_within_1e_6=all(row["matches_expected_within_1e_6"] for row in results))
    write_json(output / "replay_manifest.json", manifest)
    write_json(output / "replay_results.json", {"manifest": manifest, "folds": results})
    print(f"COMPLETE: {len(results)} folds, {manifest['seconds']:.1f}s; "
          f"matches expected={manifest['matches_expected_within_1e_6']}", flush=True)


if __name__ == "__main__":
    main()
