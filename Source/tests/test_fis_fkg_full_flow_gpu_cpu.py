#!/usr/bin/env python3
"""
Run full FIS + FKG flow on CPU and GPU, then compare speed/accuracy.

Datasets:
- ICTA (single CSV split)
- Diabetic Retinopathy Feature FT Selection (train/test CSV)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_windows_dll_dirs() -> None:
    if os.name != "nt":
        return

    candidates = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")

    for path in candidates:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass


def resolve_module_dir(module_dir: str) -> Path:
    root = project_root()
    candidates = []
    if module_dir in ("gpu", "auto"):
        candidates.append(root / "GPU" / "Source")
    if module_dir in ("source", "auto"):
        candidates.append(root / "Source")

    for path in candidates:
        if any(path.glob("fisa_module*.pyd")):
            return path
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not resolve module dir for fisa_module.")


def normalize_label(value):
    fv = float(value)
    iv = int(round(fv))
    if abs(fv - float(iv)) < 1e-9:
        return iv
    return fv


def load_numeric_frame(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    drop_cols = [col for col in df.columns if str(col).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def split_train_test(df, test_ratio: float, seed: int):
    if len(df) < 4:
        raise ValueError("Dataset is too small for train/test split.")
    indices = list(range(len(df)))
    random.Random(seed).shuffle(indices)
    split = int(len(df) * (1.0 - test_ratio))
    split = max(2, min(split, len(df) - 2))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def accuracy(predicted: List, actual: List) -> float:
    if not actual:
        return 0.0
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    return correct * 100.0 / len(actual)


def remap_fis_predictions(train_pred_clusters: List[int],
                          train_true_labels: List,
                          test_pred_clusters: List[int]) -> List:
    bucket: Dict[int, Counter] = defaultdict(Counter)
    for cluster_id, true_label in zip(train_pred_clusters, train_true_labels):
        bucket[int(cluster_id)][true_label] += 1

    if train_true_labels:
        global_major = Counter(train_true_labels).most_common(1)[0][0]
    else:
        global_major = 0

    cluster_to_label = {}
    for cluster_id, counter in bucket.items():
        cluster_to_label[cluster_id] = counter.most_common(1)[0][0]

    remapped = []
    for cluster_id in test_pred_clusters:
        remapped.append(cluster_to_label.get(int(cluster_id), global_major))
    return remapped


def label_index_maps(train_labels: List, test_labels: List):
    labels = train_labels + test_labels
    unique_labels = sorted(set(labels), key=lambda x: (isinstance(x, float), x))
    label_to_idx = {label: idx + 1 for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx + 1: label for idx, label in enumerate(unique_labels)}
    return label_to_idx, idx_to_label


def gauss_mf(x: float, sigma: float, center: float) -> float:
    if sigma <= 0:
        sigma = 1e-10
    diff = x - center
    return pow(2.718281828459045, -(diff * diff) / (2.0 * sigma * sigma))


def fuzzify_inputs_with_fis(fis, inputs: List[List[float]]) -> List[List[int]]:
    centers = fis.get_centers()
    sigma = fis.get_sigma()
    rules = []

    for row in inputs:
        fuzzy_row = []
        for feature_idx, value in enumerate(row):
            center_vector = centers[feature_idx]
            feature_sigma = float(sigma[feature_idx])
            memberships = [
                gauss_mf(float(value), feature_sigma, float(center))
                for center in center_vector
            ]
            fuzzy_row.append(int(max(range(len(memberships)), key=memberships.__getitem__)) + 1)
        rules.append(fuzzy_row)

    return rules


def make_fis_rule_records(
    fis,
    train_inputs: List[List[float]],
    train_labels: List,
    test_inputs: List[List[float]],
    test_labels: List,
):
    label_to_idx, idx_to_label = label_index_maps(train_labels, test_labels)
    train_fuzzy = fuzzify_inputs_with_fis(fis, train_inputs)
    test_fuzzy = fuzzify_inputs_with_fis(fis, test_inputs)

    train_records = [
        fuzzy_row + [int(label_to_idx[label])]
        for fuzzy_row, label in zip(train_fuzzy, train_labels)
    ]
    test_records = [
        fuzzy_row + [int(label_to_idx[label])]
        for fuzzy_row, label in zip(test_fuzzy, test_labels)
    ]
    return train_records, test_records, label_to_idx, idx_to_label


def discretize_for_fkg(train_df, test_df, label_col: str, bins: int):
    import pandas as pd

    feature_cols = [c for c in train_df.columns if c != label_col]
    train_binned = pd.DataFrame(index=train_df.index)
    test_binned = pd.DataFrame(index=test_df.index)

    for col in feature_cols:
        combined = pd.concat([train_df[col], test_df[col]], ignore_index=True)
        unique_count = int(combined.nunique(dropna=False))
        if unique_count <= 1:
            train_vals = [1] * len(train_df)
            test_vals = [1] * len(test_df)
        else:
            q = min(max(2, bins), unique_count)
            ranked = combined.rank(method="first")
            try:
                binned = (pd.qcut(ranked, q=q, labels=False, duplicates="drop") + 1).astype("int64")
            except ValueError:
                min_v = float(combined.min())
                max_v = float(combined.max())
                if max_v <= min_v:
                    binned = pd.Series([1] * len(combined), dtype="int64")
                else:
                    scaled = ((combined - min_v) / (max_v - min_v) * (q - 1)).round().astype("int64") + 1
                    binned = scaled.clip(1, q)
            train_vals = binned.iloc[: len(train_df)].tolist()
            test_vals = binned.iloc[len(train_df) :].tolist()

        train_binned[col] = train_vals
        test_binned[col] = test_vals

    combined_labels = [normalize_label(v) for v in train_df[label_col].tolist() + test_df[label_col].tolist()]
    unique_labels = sorted(set(combined_labels), key=lambda x: (isinstance(x, float), x))
    label_to_idx = {label: idx + 1 for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx + 1: label for idx, label in enumerate(unique_labels)}

    train_records = []
    for i in range(len(train_df)):
        feat = [int(train_binned.iloc[i][c]) for c in feature_cols]
        label = normalize_label(train_df.iloc[i][label_col])
        train_records.append(feat + [int(label_to_idx[label])])

    test_records = []
    for i in range(len(test_df)):
        feat = [int(test_binned.iloc[i][c]) for c in feature_cols]
        label = normalize_label(test_df.iloc[i][label_col])
        test_records.append(feat + [int(label_to_idx[label])])

    return train_records, test_records, label_to_idx, idx_to_label


def run_fkg_records(
    fisa_module,
    train_records: List[List[int]],
    test_records: List[List[int]],
    idx_to_label: Dict[int, object],
    use_gpu: bool,
    warm_repeats: int,
    input_source: str,
):
    n_classes = len(idx_to_label)
    fkg = fisa_module.fkg.FKG()
    set_backend(fkg, use_gpu)

    t2 = time.perf_counter()
    fkg.train(train_records, n_classes)
    fkg_train_cold_ms = (time.perf_counter() - t2) * 1000.0

    fkg_train_warm_runs = []
    for _ in range(warm_repeats):
        tw = time.perf_counter()
        fkg.train(train_records, n_classes)
        fkg_train_warm_runs.append((time.perf_counter() - tw) * 1000.0)
    fkg_train_warm_avg_ms = (
        sum(fkg_train_warm_runs) / len(fkg_train_warm_runs)
        if fkg_train_warm_runs else 0.0
    )

    test_features = [row[:-1] for row in test_records]
    t3 = time.perf_counter()
    if hasattr(fkg, "predict_batch_with_confidence"):
        fkg_batch = fkg.predict_batch_with_confidence(test_features)
        fkg_pred_classes = [int(item[0]) for item in fkg_batch]
    elif hasattr(fkg, "predict_batch"):
        fkg_pred_classes = [int(v) for v in fkg.predict_batch(test_features)]
    else:
        fkg_pred_classes = [int(fkg.predict(row)[0]) for row in test_features]
    fkg_infer_cold_ms = (time.perf_counter() - t3) * 1000.0

    fkg_infer_warm_runs = []
    for _ in range(warm_repeats):
        tw = time.perf_counter()
        if hasattr(fkg, "predict_batch_with_confidence"):
            _ = fkg.predict_batch_with_confidence(test_features)
        elif hasattr(fkg, "predict_batch"):
            _ = fkg.predict_batch(test_features)
        else:
            _ = [fkg.predict(row)[0] for row in test_features]
        fkg_infer_warm_runs.append((time.perf_counter() - tw) * 1000.0)
    fkg_infer_warm_avg_ms = (
        sum(fkg_infer_warm_runs) / len(fkg_infer_warm_runs)
        if fkg_infer_warm_runs else 0.0
    )

    fkg_pred_labels = [idx_to_label.get(cls, cls) for cls in fkg_pred_classes]
    fkg_true_labels = [idx_to_label[int(row[-1])] for row in test_records]
    fkg_acc = accuracy(fkg_pred_labels, fkg_true_labels)
    fkg_using_gpu = bool(fkg.is_using_gpu()) if hasattr(fkg, "is_using_gpu") else use_gpu

    unique_train_rules = len({tuple(row) for row in train_records})

    return {
        "effective_backend": "gpu" if fkg_using_gpu else "cpu",
        "input_source": input_source,
        "train_rule_count": len(train_records),
        "test_rule_count": len(test_records),
        "unique_train_rules": unique_train_rules,
        "duplicate_train_rules": len(train_records) - unique_train_rules,
        "train_ms": {
            "cold": fkg_train_cold_ms,
            "warm_avg": fkg_train_warm_avg_ms,
            "warm_runs": fkg_train_warm_runs,
        },
        "infer_ms": {
            "cold_total": fkg_infer_cold_ms,
            "cold_per_sample": fkg_infer_cold_ms / max(1, len(test_features)),
            "warm_avg_total": fkg_infer_warm_avg_ms,
            "warm_avg_per_sample": fkg_infer_warm_avg_ms / max(1, len(test_features)),
            "warm_runs": fkg_infer_warm_runs,
        },
        "accuracy_pct": fkg_acc,
        "pred_first10": fkg_pred_labels[:10],
        "_fkg_predictions": fkg_pred_labels,
    }


def set_backend(instance, use_gpu: bool) -> None:
    if hasattr(instance, "set_use_gpu"):
        instance.set_use_gpu(use_gpu)
    elif hasattr(instance, "setUseGPU"):
        instance.setUseGPU(use_gpu)


def run_one_backend(
    fisa_module,
    train_df,
    test_df,
    label_col: str,
    bins: int,
    use_gpu: bool,
    warm_repeats: int,
    feature_cluster_count: int = 3,
    include_raw_fkg: bool = False,
):
    backend_name = "gpu" if use_gpu else "cpu"
    warm_repeats = max(0, int(warm_repeats))

    # -------------------------
    # FIS stage
    # -------------------------
    train_labels = [normalize_label(v) for v in train_df[label_col].tolist()]
    test_labels = [normalize_label(v) for v in test_df[label_col].tolist()]

    train_matrix = train_df.values.tolist()
    train_matrix = [[float(v) for v in row] for row in train_matrix]
    test_inputs = test_df.drop(columns=[label_col]).values.tolist()
    test_inputs = [[float(v) for v in row] for row in test_inputs]
    train_inputs = train_df.drop(columns=[label_col]).values.tolist()
    train_inputs = [[float(v) for v in row] for row in train_inputs]

    n_features = len(test_inputs[0]) if test_inputs else (len(train_inputs[0]) if train_inputs else 0)
    class_count = max(2, len(set(train_labels)))
    clusters = [int(feature_cluster_count)] * n_features + [class_count]

    fis = fisa_module.fis.FIS(clusters, 2.0, 1e-5, 200)
    set_backend(fis, use_gpu)

    t0 = time.perf_counter()
    fis.train(train_matrix)
    fis_train_cold_ms = (time.perf_counter() - t0) * 1000.0

    fis_train_warm_runs = []
    for _ in range(warm_repeats):
        tw = time.perf_counter()
        fis.train(train_matrix)
        fis_train_warm_runs.append((time.perf_counter() - tw) * 1000.0)
    fis_train_warm_avg_ms = (
        sum(fis_train_warm_runs) / len(fis_train_warm_runs)
        if fis_train_warm_runs else 0.0
    )

    t1 = time.perf_counter()
    test_pred_clusters = [int(v) for v in fis.predict_batch(test_inputs)]
    fis_infer_cold_ms = (time.perf_counter() - t1) * 1000.0

    fis_infer_warm_runs = []
    for _ in range(warm_repeats):
        tw = time.perf_counter()
        _ = [int(v) for v in fis.predict_batch(test_inputs)]
        fis_infer_warm_runs.append((time.perf_counter() - tw) * 1000.0)
    fis_infer_warm_avg_ms = (
        sum(fis_infer_warm_runs) / len(fis_infer_warm_runs)
        if fis_infer_warm_runs else 0.0
    )

    train_pred_clusters = [int(v) for v in fis.predict_batch(train_inputs)]
    fis_pred_labels = remap_fis_predictions(train_pred_clusters, train_labels, test_pred_clusters)
    fis_acc = accuracy(fis_pred_labels, test_labels)

    # -------------------------
    # FKG stage
    # -------------------------
    fis_rule_train_records, fis_rule_test_records, _fis_label_to_idx, fis_idx_to_label = make_fis_rule_records(
        fis, train_inputs, train_labels, test_inputs, test_labels
    )
    fkg_result = run_fkg_records(
        fisa_module,
        fis_rule_train_records,
        fis_rule_test_records,
        fis_idx_to_label,
        use_gpu=use_gpu,
        warm_repeats=warm_repeats,
        input_source="fis_membership_rules",
    )

    raw_fkg_result = None
    if include_raw_fkg:
        train_records, test_records, _label_to_idx, raw_idx_to_label = discretize_for_fkg(
            train_df, test_df, label_col=label_col, bins=bins
        )
        raw_fkg_result = run_fkg_records(
            fisa_module,
            train_records,
            test_records,
            raw_idx_to_label,
            use_gpu=use_gpu,
            warm_repeats=warm_repeats,
            input_source=f"raw_quantile_discretized_bins_{bins}",
        )

    fis_using_gpu = bool(fis.is_using_gpu()) if hasattr(fis, "is_using_gpu") else use_gpu

    result = {
        "requested_backend": backend_name,
        "effective_backend": {
            "fis": "gpu" if fis_using_gpu else "cpu",
            "fkg": fkg_result["effective_backend"],
        },
        "fis": {
            "feature_cluster_count": int(feature_cluster_count),
            "train_ms": {
                "cold": fis_train_cold_ms,
                "warm_avg": fis_train_warm_avg_ms,
                "warm_runs": fis_train_warm_runs,
            },
            "infer_ms": {
                "cold_total": fis_infer_cold_ms,
                "cold_per_sample": fis_infer_cold_ms / max(1, len(test_inputs)),
                "warm_avg_total": fis_infer_warm_avg_ms,
                "warm_avg_per_sample": fis_infer_warm_avg_ms / max(1, len(test_inputs)),
                "warm_runs": fis_infer_warm_runs,
            },
            "accuracy_pct": fis_acc,
            "pred_first10": fis_pred_labels[:10],
        },
        "fkg": fkg_result,
        "test_size": len(test_inputs),
        "_fis_predictions": fis_pred_labels,
        "_fkg_predictions": fkg_result["_fkg_predictions"],
    }
    if raw_fkg_result is not None:
        result["fkg_raw_discretized"] = raw_fkg_result
        result["_fkg_raw_predictions"] = raw_fkg_result["_fkg_predictions"]
    return result


def dataset_icta(args):
    csv_path = Path(args.icta_csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"ICTA dataset not found: {csv_path}")
    df = load_numeric_frame(csv_path)
    label_col = df.columns[-1]
    train_df, test_df = split_train_test(df, args.test_ratio, args.seed)
    return {
        "name": "icta",
        "train_df": train_df,
        "test_df": test_df,
        "label_col": label_col,
        "source": str(csv_path),
        "feature_cluster_count": 3,
    }


def dataset_feature_selection(args):
    train_path = Path(args.feature_train_csv).resolve()
    test_path = Path(args.feature_test_csv).resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Feature selection train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Feature selection test file not found: {test_path}")

    train_df = load_numeric_frame(train_path)
    test_df = load_numeric_frame(test_path)
    label_col = train_df.columns[-1]

    # Align columns if train/test come from different export paths.
    if list(train_df.columns) != list(test_df.columns):
        common = [c for c in train_df.columns if c in test_df.columns]
        train_df = train_df[common]
        test_df = test_df[common]
        label_col = train_df.columns[-1]

    return {
        "name": "feature_selection",
        "train_df": train_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "label_col": label_col,
        "source": f"{train_path} | {test_path}",
        "feature_cluster_count": 5,
        "feature_cluster_source": "Source_code/main/diabetic_retinopathy/Scenario_diabetic_retinopathy_fusion_feature_with_ft_selection.py",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Full flow benchmark for FIS + FKG (CPU/GPU).")
    parser.add_argument("--dataset", choices=["icta", "feature_selection", "both"], default="both")
    parser.add_argument("--module-dir", choices=["auto", "source", "gpu"], default="source")
    parser.add_argument(
        "--backend",
        choices=["both", "cpu", "gpu"],
        default="both",
        help="Backend set to run. Use gpu for large FKG datasets; CPU FKG is exhaustive and slow.",
    )
    parser.add_argument("--bins", type=int, default=6, help="Bins for FKG discretization.")
    parser.add_argument(
        "--include-raw-fkg",
        action="store_true",
        help="Also run the slow diagnostic FKG path that discretizes raw features directly.",
    )
    parser.add_argument("--warm-repeats", type=int, default=1,
                        help="Number of warm runs after cold run (per train/infer stage).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--icta-csv", default=str(project_root() / "Source_code" / "data" / "ICTA" / "ICTA.csv"))
    parser.add_argument(
        "--feature-train-csv",
        default=str(project_root() / "Source_code" / "data" / "FIS" / "input" /
                    "Diabetic Retinopathy Feature FT Selection" / "train_data.csv"),
    )
    parser.add_argument(
        "--feature-test-csv",
        default=str(project_root() / "Source_code" / "data" / "FIS" / "input" /
                    "Diabetic Retinopathy Feature FT Selection" / "test_data.csv"),
    )
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    module_dir = resolve_module_dir(args.module_dir)
    add_windows_dll_dirs()
    sys.path.insert(0, str(module_dir))

    import fisa_module  # noqa: PLC0415

    gpu_compiled = bool(getattr(fisa_module, "GPU_COMPILED", False))
    gpu_available = bool(fisa_module.is_gpu_available() if hasattr(fisa_module, "is_gpu_available") else False)

    datasets = []
    if args.dataset in ("icta", "both"):
        datasets.append(dataset_icta(args))
    if args.dataset in ("feature_selection", "both"):
        datasets.append(dataset_feature_selection(args))

    report = {
        "module_dir": str(module_dir),
        "gpu_compiled": gpu_compiled,
        "gpu_available": gpu_available,
        "datasets": {},
    }

    for ds in datasets:
        train_df = ds["train_df"]
        test_df = ds["test_df"]
        label_col = ds["label_col"]
        feature_cluster_count = int(ds.get("feature_cluster_count", 3))

        backend_results = {}
        if args.backend in ("both", "cpu"):
            backend_results["cpu"] = run_one_backend(
                fisa_module, train_df, test_df, label_col, args.bins, use_gpu=False,
                warm_repeats=args.warm_repeats,
                feature_cluster_count=feature_cluster_count,
                include_raw_fkg=args.include_raw_fkg,
            )
        if args.backend in ("both", "gpu"):
            backend_results["gpu"] = run_one_backend(
                fisa_module, train_df, test_df, label_col, args.bins, use_gpu=True,
                warm_repeats=args.warm_repeats,
                feature_cluster_count=feature_cluster_count,
                include_raw_fkg=args.include_raw_fkg,
            )

        cpu_gpu_match_pct = {}
        if "cpu" in backend_results and "gpu" in backend_results:
            cpu_result = backend_results["cpu"]
            gpu_result = backend_results["gpu"]
            cpu_gpu_match_pct["fis"] = accuracy(
                gpu_result["_fis_predictions"], cpu_result["_fis_predictions"]
            )
            cpu_gpu_match_pct["fkg"] = accuracy(
                gpu_result["_fkg_predictions"], cpu_result["_fkg_predictions"]
            )
            if "_fkg_raw_predictions" in cpu_result and "_fkg_raw_predictions" in gpu_result:
                cpu_gpu_match_pct["fkg_raw_discretized"] = accuracy(
                    gpu_result["_fkg_raw_predictions"], cpu_result["_fkg_raw_predictions"]
                )

        for result in backend_results.values():
            result.pop("_fis_predictions", None)
            result.pop("_fkg_predictions", None)
            result.pop("_fkg_raw_predictions", None)
            result.get("fkg", {}).pop("_fkg_predictions", None)
            result.get("fkg_raw_discretized", {}).pop("_fkg_predictions", None)

        dataset_report = {
            "source": ds["source"],
            "label_column": label_col,
            "feature_cluster_count": feature_cluster_count,
            "feature_cluster_source": ds.get("feature_cluster_source", "default"),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "cpu_gpu_match_pct": cpu_gpu_match_pct,
        }
        dataset_report.update(backend_results)
        report["datasets"][ds["name"]] = dataset_report

    print("=" * 100)
    print("FIS + FKG Full Flow CPU/GPU Benchmark")
    print("=" * 100)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 100)

    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Saved report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
