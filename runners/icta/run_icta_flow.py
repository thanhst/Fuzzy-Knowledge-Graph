#!/usr/bin/env python3
"""Clean ICTA runner that calls reusable flow scripts from Source/Src/Flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_step(label: str, args: list[str]) -> None:
    print(f"[STEP] {label}")
    subprocess.check_call(args, cwd=repo_root())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ICTA processing + FKG flow.")
    parser.add_argument("--under5", action="store_true", help="Use the under-5 image flow.")
    parser.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-tab", type=int, default=8)
    parser.add_argument("--k-img", type=int, default=5)
    parser.add_argument("--under5-max-images", type=int, default=5)
    parser.add_argument("--under5-image-size", type=int, default=512)
    args = parser.parse_args()

    root = repo_root()
    source = root / "Source"
    flow = source / "Src" / "Flow"
    py = sys.executable

    run_step("1/6 process table", [py, str(flow / "process_table.py")])

    if args.under5:
        run_step(
            "2/6 process images under5",
            [
                py,
                str(flow / "process_image_under5.py"),
                "--max-images",
                str(args.under5_max_images),
                "--image-size",
                str(args.under5_image_size),
            ],
        )
        image_dir = source / "Data" / "processing" / "image_under5"
        fusion_dir = source / "Data" / "processing" / "fusion_under5"
        result_dir = source / "Data" / "result" / "ICTA_under5"
    else:
        run_step("2/6 process images", [py, str(flow / "process_image.py")])
        image_dir = source / "Data" / "processing" / "image"
        fusion_dir = source / "Data" / "processing" / "fusion"
        result_dir = source / "Data" / "result" / "ICTA"

    run_step("3/6 select table features", [py, str(flow / "select_table_features.py"), "--k", str(args.k_tab)])
    run_step(
        "4/6 select image features",
        [
            py,
            str(flow / "select_image_features.py"),
            "--input-csv",
            str(image_dir / "image_features_patient.csv"),
            "--out-dir",
            str(image_dir),
            "--k",
            str(args.k_img),
        ],
    )
    run_step(
        "5/6 fusion",
        [
            py,
            str(flow / "fusion.py"),
            "--table-csv",
            str(source / "Data" / "processing" / "table" / "table_features_selected.csv"),
            "--image-csv",
            str(image_dir / "image_features_selected.csv"),
            "--out-dir",
            str(fusion_dir),
        ],
    )
    run_step(
        "6/6 FKG",
        [
            py,
            str(flow / "run_fkg_gpu_flow.py"),
            "--fusion-csv",
            str(fusion_dir / "fusion_selected.csv"),
            "--out-dir",
            str(result_dir),
            "--backend",
            args.backend,
            "--bins",
            str(args.bins),
            "--test-ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
        ],
    )

    print(f"[OK] Result folder: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
