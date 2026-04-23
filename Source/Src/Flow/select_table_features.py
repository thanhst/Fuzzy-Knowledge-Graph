#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def select_table_features(input_csv: Path, output_dir: Path, k: int = 8) -> Dict[str, Path]:
    df = pd.read_csv(input_csv, dtype={"patient_id": str})
    required_cols = {"patient_id", "Outcome"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input table CSV must contain columns: {required_cols}")

    feature_cols = [c for c in df.columns if c not in {"patient_id", "Outcome"}]
    if not feature_cols:
        raise ValueError("No table feature columns found to select.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df["Outcome"], errors="coerce").fillna(0).astype(int)

    actual_k = max(1, min(int(k), X.shape[1]))
    selector = SelectKBest(score_func=mutual_info_classif, k=actual_k)
    X_new = selector.fit_transform(X, y)
    selected_cols: List[str] = [col for col, keep in zip(feature_cols, selector.get_support()) if keep]

    selected_df = pd.DataFrame(X_new, columns=selected_cols)
    selected_df.insert(0, "patient_id", df["patient_id"].astype(str))
    selected_df["Outcome"] = y.values

    scores_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "score": selector.scores_,
            "selected": selector.get_support(),
        }
    ).sort_values(["selected", "score"], ascending=[False, False])

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_csv = output_dir / "table_features_selected.csv"
    scores_csv = output_dir / "table_feature_scores.csv"
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8")
    scores_df.to_csv(scores_csv, index=False, encoding="utf-8")

    return {"selected_csv": selected_csv, "scores_csv": scores_csv}


def main() -> int:
    parser = argparse.ArgumentParser(description="K-selection for processed table features.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=source_root() / "Data" / "processing" / "table" / "ICTA_table.csv",
        help="Input table CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=source_root() / "Data" / "processing" / "table",
        help="Output directory.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of table features to keep.")
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    outputs = select_table_features(args.input_csv, args.out_dir, k=args.k)
    print(f"[OK] Selected table features: {outputs['selected_csv']}")
    print(f"[OK] Table feature scores: {outputs['scores_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
