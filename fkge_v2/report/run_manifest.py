import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

import config as C


def create_manifest(run_id, quick, requested_experiments, argv):
    return {
        "run_id": run_id,
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "quick": quick,
        "requested_experiments": requested_experiments,
        "argv": argv,
        "source_revision": _git_revision(),
        "source_dirty": _git_dirty(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": _version("numpy"),
            "scikit_learn": _version("scikit-learn"),
            "xgboost": _version("xgboost"),
            "pykeen": _version("pykeen"),
        },
        "configuration": {
            "epochs": C.FKGE.epochs,
            "n_seeds": C.EVAL.N_SEEDS,
            "k_fold": C.EVAL.K_FOLD,
            "embedding_dimension": C.FKGE.d,
            "window": C.FKGE.w,
            "negative_samples": C.FKGE.K_neg,
            "lambda_node": C.FKGE.lam_node,
            "lambda_sgns": C.FKGE.beta_rule,
            "lambda_inf": C.FKGE.gamma_inf,
            "lambda_pred": C.FKGE.delta_pred,
            "weight_decay": C.FKGE.weight_decay,
            "aggregation": C.FKGE.aggregation,
            "max_pairs_per_epoch": C.FKGE.max_pairs_per_epoch,
            "brset_primary_modality": C.BRSET_PRIMARY_MODALITY,
        },
        "data_sources": {
            "brset_frb_package": C.PATHS.BRSET_FRB_PACKAGE,
            "diabetes_raw": C.PATHS.DIABETES_KAGGLE_RAW_FILE,
            "healthcare_diabetes_raw": C.PATHS.HEALTHCARE_DIABETES_RAW_FILE,
        },
    }


def finish_manifest(manifest, elapsed_seconds, status="completed", error=None):
    manifest["status"] = status
    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["elapsed_seconds"] = elapsed_seconds
    if error is not None:
        manifest["error"] = str(error)
    return manifest


def write_manifest(manifest):
    path = os.path.join(C.PATHS.OUTPUT_DIR, "run_manifest.json")
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return path


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision():
    repository = C.PATHS.REPOSITORY_ROOT.replace("\\", "/")
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
            cwd=C.PATHS.REPOSITORY_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty():
    repository = C.PATHS.REPOSITORY_ROOT.replace("\\", "/")
    try:
        return bool(subprocess.check_output(
            ["git", "-c", f"safe.directory={repository}", "status", "--porcelain",
             "--untracked-files=no", "--", "fkge_v2"],
            cwd=C.PATHS.REPOSITORY_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return None
