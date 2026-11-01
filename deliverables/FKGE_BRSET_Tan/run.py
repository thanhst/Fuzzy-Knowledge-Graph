"""Standalone full-BRSET FKG-E with patient-held-out validation and reports."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import pickle
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from build import IDS, IMAGE_COLUMNS, TABLE_COLUMNS, TARGET, verify_release
from compact_fkge import FCMFuzzifier, compress_rules
from portable_folds import create_assignments, verify_assignments
from preprocess import FusionPreprocessor, balance_teacher_parts
from train_fkge import CompactFKGE, build_teacher

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=lambda item: item.tolist()) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hardware(device):
    info = {"system": platform.platform(), "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(), "device": device,
            "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(gpu_name=props.name, gpu_memory_bytes=props.total_memory)
    try:
        import psutil
        info["ram_bytes"] = psutil.virtual_memory().total
    except ImportError:
        pass
    return info


def repository_state():
    try:
        subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", str(ROOT / "run.py")],
                                stderr=subprocess.DEVNULL)
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(ROOT)],
                                         stderr=subprocess.DEVNULL, text=True).strip()
        return {"commit": commit, "package_dirty": bool(status)}
    except (FileNotFoundError, subprocess.CalledProcessError):
        release = ROOT / "RELEASE.json"
        if release.exists():
            info = read_json(release)
            info["package_dirty"] = any(not (ROOT / name).is_file() or sha(ROOT / name) != expected
                                        for name, expected in info["sha256"].items())
            return info
        return {"commit": None, "source_hashes_recorded": True}


def synthetic_data(seed):
    """Small fixture only; never sampled from the real BRSET outer test."""
    rng = np.random.RandomState(seed)
    patients = np.repeat(np.arange(100), 2)
    y = (patients % 4 == 0).astype(int)
    data = pd.DataFrame({"image_id": [f"smoke_{i:04}" for i in range(len(y))],
                         "patient_id": [f"smoke_p{p:03}" for p in patients], TARGET: y})
    for column in [*TABLE_COLUMNS, *IMAGE_COLUMNS]:
        data[column] = rng.normal(size=len(y)) + 0.4 * y
    data.loc[::9, "patient_age"] = np.nan
    assignments, _ = create_assignments(data[IDS], TARGET, folds=5, seed=seed)
    return data, assignments


def prepare_fold(data, assigned, fold, directory, seed, config):
    if (directory / "preparation.json").exists():
        return read_json(directory / "preparation.json")
    test_ids = set(assigned.loc[assigned.fold == fold, "image_id"])
    pool = data.loc[~data.image_id.isin(test_ids)].reset_index(drop=True)
    test = data.loc[data.image_id.isin(test_ids)].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + fold)
    ti, vi = next(splitter.split(np.zeros((len(pool), 1)), pool[TARGET], pool.patient_id))
    train, val = pool.iloc[ti].reset_index(drop=True), pool.iloc[vi].reset_index(drop=True)
    sets = [set(frame.patient_id) for frame in (train, val, test)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise AssertionError("Patients overlap between inner train, validation and outer test")
    if any(frame[TARGET].nunique() != 2 for frame in (train, val, test)):
        raise ValueError("Every split must contain both classes")
    processing = FusionPreprocessor(config["k_img"], config["k_tab"]).fit(train, train[TARGET])
    X = processing.transform(train)
    balanced, y, patients, parents, part_full = balance_teacher_parts(X, train[TARGET], train.patient_id, seed + fold)
    fuzzifier = FCMFuzzifier(5).fit(balanced)
    R = fuzzifier.transform(balanced)
    compact = compress_rules(R, y, patients, parents, merge_signatures=False)
    # Computational teacher partitions are never statistical replicates.
    unit_parts = pd.DataFrame({"unit": compact["original_units"], "part": part_full})
    if (unit_parts.groupby("unit").part.nunique() != 1).any():
        raise AssertionError("A real patient or a SMOTE parent link crosses teacher partitions")
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "prepared.npz", R=compact["R"], y=compact["y"],
                        weights=compact["weights"], units=compact["units"],
                        parts=part_full[compact["index"]],
                        Rval=fuzzifier.transform(processing.transform(val)), yval=val[TARGET].to_numpy(int),
                        Rtest=fuzzifier.transform(processing.transform(test)), ytest=test[TARGET].to_numpy(int))
    for name, frame in (("train_original", train), ("validation", val), ("test", test)):
        frame[IDS].to_csv(directory / f"{name}_ids.csv", index=False)
    train[IDS].assign(teacher_part=part_full[:len(train)]).to_csv(directory / "teacher_train_ids.csv", index=False)
    pd.DataFrame({"synthetic_row": np.arange(len(train), len(y)),
                  "left_image_id": train.image_id.to_numpy()[parents[:, 0]],
                  "right_image_id": train.image_id.to_numpy()[parents[:, 1]]}).to_csv(
                      directory / "smote_parents.csv", index=False)
    with (directory / "preprocessing.pkl").open("wb") as stream:
        pickle.dump((processing, fuzzifier), stream)
    info = {"fold": fold, "outer_train_images": len(pool), "outer_train_patients": pool.patient_id.nunique(),
            "inner_train_images": len(train), "inner_train_patients": train.patient_id.nunique(),
            "validation_images": len(val), "validation_patients": val.patient_id.nunique(),
            "test_images": len(test), "test_patients": test.patient_id.nunique(),
            "rules_including_smote": len(y), "smote_rows": len(parents), "unique_rules": len(compact["R"]),
            "teacher_units": len(np.unique(compact["units"])),
            "largest_unit_rows": int(np.bincount(compact["original_units"]).max()),
            "teacher_part_rows": np.bincount(part_full, minlength=5),
            "selected_features": processing.feature_names, "patient_overlap_count": 0,
            "inner_split_and_smote_seed": seed + fold, "training_seed": seed,
            "fit_population": "inner_train_only", "smote_scope": "within_each_inner_training_teacher_partition",
            "teacher_units_are_independent_samples": False}
    write_json(directory / "preparation.json", info)
    return info


def train_fold(directory, config, seed, device, block_size, cache):
    if (directory / "results.json").exists():
        return read_json(directory / "results.json")
    with np.load(directory / "prepared.npz") as archive:
        prepared = {name: archive[name] for name in archive.files if name not in {"Rtest", "ytest"}}
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "teacher_complete.json").exists():
        teacher, elapsed = build_teacher(prepared, cache)
        write_json(cache / "teacher_complete.json", {"seconds": elapsed})
        del teacher
    teacher_small = dict(np.load(cache / "teacher_small.npz"))
    attribution = np.load(cache / "teacher_attribution.npy", mmap_mode="r")

    def make_model(parameters):
        return CompactFKGE(prepared["R"], prepared["y"], prepared["weights"], prepared["units"],
                           teacher_small["pF"], attribution, teacher_small["yF"],
                           seed=seed, device=device, block_size=block_size, **parameters)

    names = list(config["grid"])
    trials = [dict(zip(names, values)) for values in itertools.product(*(config["grid"][name] for name in names))]
    records = []
    for index, parameters in enumerate(trials, 1):
        path = directory / "trials" / f"trial_{index:03}.json"
        if path.exists():
            record = read_json(path)
            if record["parameters"] != parameters:
                raise ValueError("Trial configuration differs from cached trial")
        else:
            print(f"START {directory.name} trial {index}/{len(trials)}: {parameters}", flush=True)
            model = make_model(parameters).fit(prepared["Rval"], prepared["yval"],
                                               max_epochs=config["max_epochs"], patience=config["patience"], verbose=True)
            record = {"trial": index, "parameters": parameters, "validation_log_loss": model.val_loss,
                      "threshold": model.threshold, "epochs": model.epochs, "seconds": model.seconds,
                      "stop_reason": model.stop_reason, "history": model.history}
            if not np.isfinite(model.val_loss):
                raise FloatingPointError("Nonfinite validation loss; the trial cannot be selected")
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(path.with_suffix(".npz"))
            write_json(path, record)
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        records.append(record)
        print(f"{directory.name}: trial {index}/{len(trials)}, validation loss={record['validation_log_loss']:.6f}", flush=True)
    chosen = min(records, key=lambda row: row["validation_log_loss"])
    # Lock the selection BEFORE loading outer-test features or labels for evaluation.
    write_json(directory / "selected.json", {"criterion": "minimum_real_validation_log_loss", **chosen})
    model = make_model(chosen["parameters"])
    checkpoint = np.load(directory / "trials" / f"trial_{chosen['trial']:03}.npz")
    model.params = [model.tensor(checkpoint[name]).requires_grad_(True) for name in ("Z", "C", "U")]
    model.threshold = float(checkpoint["threshold"])
    model.save(directory / "selected_model.npz")
    with np.load(directory / "prepared.npz") as archive:
        prepared["Rtest"], prepared["ytest"] = archive["Rtest"], archive["ytest"]
    started = time.perf_counter()
    probability = model.probabilities(prepared["Rtest"])
    prediction_seconds = time.perf_counter() - started
    score_e = np.log(probability[:, 1]) - np.log(probability[:, 0])
    from compact_fkge import CompactFISA
    teacher = CompactFISA(prepared["R"], prepared["y"], prepared["weights"])
    D, matched = teacher.log_values(prepared["Rtest"])
    score_f = np.where(matched, D[:, 1] - D[:, 0], 0.0)
    threshold_f = float(teacher_small["threshold"])
    pred_f = np.where(matched, (score_f > threshold_f).astype(int), teacher.fallback)
    ids = pd.read_csv(directory / "test_ids.csv", dtype={"image_id": str, "patient_id": str})
    ids["fisa_score"], ids["fkge_score"] = score_f, score_e
    ids["fisa_prediction"], ids["fkge_prediction"] = pred_f, (score_e > model.threshold).astype(int)
    ids.to_csv(directory / "predictions.csv", index=False)
    from report import binary_metrics
    results = {"fold": int(directory.name[-2:]), "selected_parameters": chosen["parameters"],
               "validation_log_loss": chosen["validation_log_loss"], "selected_epochs": chosen["epochs"],
               "prediction_seconds": prediction_seconds,
               "FKG-E": binary_metrics(prepared["ytest"], ids.fkge_prediction, score_e),
               "FISA": binary_metrics(prepared["ytest"], pred_f, score_f)}
    write_json(directory / "results.json", results)
    del attribution, model, teacher
    gc.collect()
    return results


def run_experiment(data, assignments, output, config, folds, seed, device, block_size, scope, stage, resume):
    verify_assignments(assignments, TARGET, 5)
    pd.testing.assert_frame_equal(data[IDS].reset_index(drop=True), assignments[IDS].reset_index(drop=True), check_dtype=False)
    code_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(ROOT.rglob("*.py"))
                   if not any(part in {"outputs", "private", "__pycache__"} for part in path.relative_to(ROOT).parts)}
    identity = {"scope": scope, "config": config, "seed": seed, "folds": folds, "device": device,
                "block_size": block_size, "code_sha256": code_hashes,
                "data_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
                "split_sha256": hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest(),
                "packages": {name: importlib.metadata.version(name) for name in
                             ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "torch", "matplotlib"]}}
    manifest_path = output / "run_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.exists() or read_json(manifest_path)["identity"] != identity:
            raise ValueError("Use a new output directory, or --resume with exactly the same data/code/config/environment")
        manifest = read_json(manifest_path)
        manifest.setdefault("resume_events", []).append({"hardware": hardware(device), "exact_argv": [sys.executable, *sys.argv]})
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"identity": identity, "repository": repository_state(), "hardware": hardware(device),
                    "exact_argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
                    "selection": "validation only; outer test after selecting all trials; no refit on validation",
                    "statistical_unit": "five outer patient-disjoint folds, never teacher partitions or blocks",
                    "status": "started"}
        write_json(manifest_path, manifest)
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output / "environment-freeze.txt").write_text(freeze, encoding="utf-8")
    cache_key = hashlib.sha256((str(output) + json.dumps(identity, sort_keys=True)).encode()).hexdigest()[:20]
    cache_root = Path(tempfile.gettempdir()) / "fkge_brset_teacher" / cache_key
    manifest["temporary_teacher_cache"] = str(cache_root)
    for fold in folds:
        directory = output / f"fold_{fold:02}"
        info = prepare_fold(data, assignments, fold, directory, seed, config)
        print(f"Prepared fold {fold}: {info['inner_train_patients']} real training patients, {info['rules_including_smote']} rules", flush=True)
        if stage != "prepare":
            train_fold(directory, config, seed, device, block_size, cache_root / directory.name)
        from report import render_report
        render_report(output, scope)
        manifest["completed_folds"] = [k for k in folds if (output / f"fold_{k:02}/results.json").exists()]
        write_json(manifest_path, manifest)
    manifest["status"] = "prepared" if stage == "prepare" else "complete"
    write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/brset_full")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--stage", choices=["all", "prepare", "report"], default="all")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Synthetic end-to-end check; never a BRSET result")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(verify_release(ROOT / "data"), indent=2))
        return
    if args.stage == "report":
        from report import render_report
        manifest = read_json(args.output.resolve() / "run_manifest.json")
        render_report(args.output.resolve(), manifest["identity"]["scope"])
        return
    if sorted(set(args.folds)) != sorted(args.folds) or not set(args.folds) <= set(range(1, 6)):
        parser.error("--folds must contain distinct numbers from 1 to 5")
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu or use a GPU runtime")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = read_json(args.config)
    expected_grid = {"alpha", "d_e", "lam_I", "lam_P"}
    if set(config["grid"]) != expected_grid or any(not values for values in config["grid"].values()):
        parser.error("Config must specify nonempty alpha, d_e, lam_I and lam_P grids")
    if config["max_epochs"] < 1 or config["patience"] < 1 or min(config["k_img"], config["k_tab"]) < 1:
        parser.error("Epochs, patience and feature counts must be positive")
    if any(not 0 <= alpha <= 1 for alpha in config["grid"]["alpha"]) or any(
        value < 0 for name in ("lam_I", "lam_P") for value in config["grid"][name]
    ) or any(not isinstance(d, int) or d < 1 for d in config["grid"]["d_e"]):
        parser.error("Invalid hyperparameter range")
    if args.smoke:
        data, assignments = synthetic_data(args.seed)
        config.update(grid={"alpha": [0.3, 0.7], "d_e": [4], "lam_I": [0.3], "lam_P": [0.5]}, max_epochs=2, patience=2)
        config.update(k_img=2, k_tab=2)
        scope = "SYNTHETIC_SMOKE_ONLY"
    else:
        verify_release(ROOT / "data")
        data = pd.read_csv(ROOT / "data/fusion_features.csv", dtype={"image_id": str, "patient_id": str}, float_precision="round_trip")
        assignments = pd.read_csv(ROOT / "data/fold_assignments.csv", dtype={"image_id": str, "patient_id": str})
        scope = "BRSET_FULL_16266_IMAGES_8524_PATIENTS"
    run_experiment(data, assignments, args.output.resolve(), config, args.folds, args.seed, device,
                   args.block_size, scope, args.stage, args.resume)
    print(f"DONE: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
