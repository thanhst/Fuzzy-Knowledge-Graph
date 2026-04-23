#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


def source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_fusion(table_csv: Path, image_csv: Path, output_dir: Path) -> Dict[str, Path]:
    table_df = pd.read_csv(table_csv, dtype={"patient_id": str})
    image_df = pd.read_csv(image_csv, dtype={"patient_id": str})

    for required in ("patient_id", "Outcome"):
        if required not in table_df.columns:
            raise ValueError(f"Table selected CSV is missing required column: {required}")
        if required not in image_df.columns:
            raise ValueError(f"Image selected CSV is missing required column: {required}")

    table_df["patient_id"] = table_df["patient_id"].astype(str)
    image_df["patient_id"] = image_df["patient_id"].astype(str)

    table_no_outcome = table_df.drop(columns=["Outcome"])
    image_no_outcome = image_df.drop(columns=["Outcome"])

    fused_df = table_no_outcome.merge(image_no_outcome, on="patient_id", how="inner", suffixes=("_tab", "_img"))
    outcome_map = dict(zip(table_df["patient_id"], table_df["Outcome"]))
    fused_df["Outcome"] = fused_df["patient_id"].map(outcome_map).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    fused_csv = output_dir / "fusion_selected.csv"
    fused_df.to_csv(fused_csv, index=False, encoding="utf-8")

    stats_df = pd.DataFrame(
        [
            {
                "table_rows": len(table_df),
                "image_rows": len(image_df),
                "fused_rows": len(fused_df),
                "table_cols": table_df.shape[1],
                "image_cols": image_df.shape[1],
                "fused_cols": fused_df.shape[1],
            }
        ]
    )
    stats_csv = output_dir / "fusion_stats.csv"
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8")
    return {"fused_csv": fused_csv, "stats_csv": stats_csv}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse selected table and image features by patient_id.")
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=source_root() / "Data" / "processing" / "table" / "table_features_selected.csv",
        help="Selected table feature CSV.",
    )
    parser.add_argument(
        "--image-csv",
        type=Path,
        default=source_root() / "Data" / "processing" / "image" / "image_features_selected.csv",
        help="Selected image feature CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=source_root() / "Data" / "processing" / "fusion",
        help="Output folder for fused data.",
    )
    args = parser.parse_args()

    if not args.table_csv.exists():
        raise FileNotFoundError(f"Table CSV not found: {args.table_csv}")
    if not args.image_csv.exists():
        raise FileNotFoundError(f"Image CSV not found: {args.image_csv}")

    outputs = run_fusion(args.table_csv, args.image_csv, args.out_dir)
    print(f"[OK] Fused dataset: {outputs['fused_csv']}")
    print(f"[OK] Fusion stats: {outputs['stats_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
