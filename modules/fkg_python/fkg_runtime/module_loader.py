"""Shared loader for the native ``fisa_module`` Python extension.

Runner scripts should import this helper instead of open-coding sys.path and
DLL search logic. The compiled extension still lives in ``Source`` or
``GPU/Source`` so old build outputs remain compatible.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class FisaModuleImport:
    module: Optional[object]
    module_dir: Optional[Path]
    error: str = ""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def candidate_module_dirs(preferred: str = "auto") -> list[Path]:
    root = repo_root()
    build_libs = sorted((root / "Source" / "build").glob("lib*")) if (root / "Source" / "build").exists() else []
    source_candidates = [
        root / "Source",
        *build_libs,
        root / "Source" / "Release",
    ]
    gpu_candidates = [
        root / "GPU" / "Source",
        root / "GPU" / "Source" / "Release",
    ]

    env_dir = os.environ.get("FISA_MODULE_DIR", "").strip()
    env_candidates = [Path(env_dir)] if env_dir else []

    if preferred == "source":
        ordered = env_candidates + source_candidates
    elif preferred == "gpu":
        ordered = env_candidates + gpu_candidates + source_candidates
    else:
        ordered = env_candidates + source_candidates + gpu_candidates

    return _dedupe(ordered)


def add_windows_dll_dirs(extra_dirs: Iterable[Path] = ()) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    root = repo_root()
    candidates: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    candidates.append(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"))
    candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")
    candidates.extend(
        [
            root / "Source",
            root / "GPU" / "Source",
            root / "GPU" / "Source" / "Release",
        ]
    )
    candidates.extend(extra_dirs)

    for path in _dedupe(candidates):
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass


def try_import_fisa_module(preferred: str = "auto", clear_existing: bool = False) -> FisaModuleImport:
    errors: list[str] = []
    candidates = candidate_module_dirs(preferred)
    add_windows_dll_dirs(candidates)

    for module_dir in candidates:
        if not module_dir.exists():
            continue
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        if clear_existing and "fisa_module" in sys.modules:
            del sys.modules["fisa_module"]
        try:
            module = importlib.import_module("fisa_module")
            return FisaModuleImport(module=module, module_dir=module_dir)
        except Exception as exc:  # pragma: no cover - import failures are environment-specific.
            errors.append(f"{module_dir}: {exc}")

    error = "\n".join(errors) if errors else "No candidate module directory found."
    return FisaModuleImport(module=None, module_dir=None, error=error)


def import_fisa_module(preferred: str = "auto", clear_existing: bool = False) -> tuple[object, Path]:
    result = try_import_fisa_module(preferred=preferred, clear_existing=clear_existing)
    if result.module is None or result.module_dir is None:
        raise ImportError(result.error)
    return result.module, result.module_dir
