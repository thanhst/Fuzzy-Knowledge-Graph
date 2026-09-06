import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEEP_MODELS = [
    ("mlp", "MLP", "Tabular"),
    ("resnet", None, "Image"),
    ("early_fusion", "Early Fusion (MLP)", "Multimodal"),
    ("late_fusion", "Late Fusion (Ensemble)", "Multimodal"),
]

FKGS_LABELS = {
    "image": ("FKG-UM (\u1ea2nh)", "Unimodal FKG"),
    "table": ("FKG-UM (B\u1ea3ng)", "Unimodal FKG"),
    "fusion": ("FKG-MM (\u0111\u1ec1 xu\u1ea5t)", "Multimodal FKG"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect diabetic retinopathy KFold deep baseline and FKG/FKGS summaries into one comparison table."
    )
    parser.add_argument(
        "--deep-summary",
        type=Path,
        required=True,
        nargs="+",
        help="One or more paths to deep baseline summary.csv files. The first summary containing a model is used for that row.",
    )
    parser.add_argument(
        "--deep-config",
        type=Path,
        default=None,
        help="Path to deep baseline config.json. Defaults to <deep-summary-dir>/config.json when present.",
    )
    parser.add_argument("--fkgs-summary", type=Path, required=True, help="Path to kfold_fkgs_mean_std_summary.csv.")
    parser.add_argument("--fkgs-tables", type=Path, default=None, help="Path to kfold_fkgs_tables.csv.")
    parser.add_argument(
        "--fkg-summary",
        type=Path,
        default=None,
        help="Optional path to native FKG kfold_modality_mean_std_summary.csv. When present, FKG rows use this full metric summary.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        required=True,
        help="Output path without extension. Writes <stem>.csv and <stem>.md.",
    )
    parser.add_argument(
        "--protocol",
        default="Patient-aware 5-fold validation; no outer test",
        help="Protocol text written to comparison rows.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_config(deep_summary: Path, deep_config: Path | None) -> dict:
    config_path = resolve_path(deep_config) if deep_config else deep_summary.parent / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in ("", None):
        return None
    return float(value)


def read_percent_fraction(row: dict[str, str], key: str) -> float | None:
    value = read_float(row, key)
    if value is None:
        return None
    return value * 100.0


def fmt_pm(mean: float | None, std: float | None, decimals: int = 1) -> str:
    if mean is None:
        return "..."
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def resnet_label(config: dict) -> str:
    arch = str(config.get("resnet_arch", "resnet18")).lower()
    if arch == "resnet50":
        return "ResNet-50"
    if arch == "resnet18":
        return "ResNet-18"
    return arch.upper()


def deep_backbone_note(rows: list[dict]) -> str:
    notes = []
    for row in rows:
        config = str(row.get("selected_config") or "")
        marker = "resnet_arch="
        if marker not in config:
            continue
        arch = config.split(marker, 1)[1].split(";", 1)[0].strip()
        notes.append(f"{row['model']}: `{arch}`")
    if not notes:
        return "L\u01b0u \u00fd: backbone \u1ea3nh c\u1ee7a t\u1eebng deep model \u0111\u01b0\u1ee3c ghi trong c\u1ed9t `Ghi ch\u00fa`."
    return "L\u01b0u \u00fd backbone \u1ea3nh: " + "; ".join(notes) + "."


def add_deep_rows(
    rows: list[dict],
    deep_sources: list[dict],
    protocol: str,
) -> None:
    for model_key, label, data_type in DEEP_MODELS:
        source = next(
            (
                candidate
                for candidate in deep_sources
                if model_key in candidate["by_model"]
            ),
            None,
        )
        if source is None:
            raise RuntimeError(f"Missing deep baseline val summary for: {model_key}")
        deep_summary = source["path"]
        config = source["config"]
        by_model = source["by_model"]
        device = config.get("device", "unknown")
        resnet_arch = str(config.get("resnet_arch", "resnet18"))
        run_final_test = bool(config.get("run_final_test", False))
        row_protocol = protocol if not run_final_test else "Patient-aware 5-fold validation plus outer test"
        run_name = deep_summary.parent.name
        model_label = resnet_label(config) if model_key == "resnet" else label
        config_note = f"runner {run_name}; device={device}"
        if model_key in {"resnet", "early_fusion", "late_fusion"}:
            config_note += f"; resnet_arch={resnet_arch}"
        row = by_model[model_key]
        rows.append(
            {
                "model": model_label,
                "data_type": data_type,
                "source_family": "deep_baseline_KFold",
                "eval_split": "val_mean_5fold",
                "protocol": row_protocol,
                "selected_config": config_note,
                "accuracy_pct": read_percent_fraction(row, "accuracy_mean"),
                "accuracy_std_pct": read_percent_fraction(row, "accuracy_std"),
                "precision_pct": read_percent_fraction(row, "precision_mean"),
                "precision_std_pct": read_percent_fraction(row, "precision_std"),
                "recall_pct": read_percent_fraction(row, "sensitivity_mean"),
                "recall_std_pct": read_percent_fraction(row, "sensitivity_std"),
                "specificity_pct": read_percent_fraction(row, "specificity_mean"),
                "specificity_std_pct": read_percent_fraction(row, "specificity_std"),
                "f1_pct": read_percent_fraction(row, "f1_mean"),
                "f1_std_pct": read_percent_fraction(row, "f1_std"),
                "auc_pct": read_percent_fraction(row, "auc_mean"),
                "auc_std_pct": read_percent_fraction(row, "auc_std"),
                "train_time_s": read_float(row, "train_seconds_mean"),
                "train_time_std_s": read_float(row, "train_seconds_std"),
                "test_time_s": read_float(row, "eval_seconds_mean"),
                "test_time_std_s": read_float(row, "eval_seconds_std"),
                "total_time_s": read_float(row, "total_time_seconds_mean"),
                "total_time_std_s": read_float(row, "total_time_seconds_std"),
                "source_path": str(deep_summary),
            }
        )


def add_fkgs_rows(
    rows: list[dict],
    fkgs_summary: Path,
    fkgs_summary_rows: list[dict[str, str]],
    protocol: str,
) -> None:
    for modality in ("image", "table", "fusion"):
        candidates = [row for row in fkgs_summary_rows if row.get("modality") == modality]
        if not candidates:
            raise RuntimeError(f"Missing FKGS summaries for modality: {modality}")
        best = max(candidates, key=lambda row: float(row["fkgs_accuracy_pct_mean"]))
        label, data_type = FKGS_LABELS[modality]
        rows.append(
            {
                "model": label,
                "data_type": data_type,
                "source_family": "FKGS_KFold",
                "eval_split": "val_mean_5fold",
                "protocol": protocol,
                "selected_config": (
                    f"best accuracy from rerun; ran={best['ran']}; "
                    f"epsilon={best['epsilon']}; folds={best['folds']}; "
                    f"features={float(best['feature_count_mean']):.0f}"
                ),
                "accuracy_pct": read_float(best, "fkgs_accuracy_pct_mean"),
                "accuracy_std_pct": read_float(best, "fkgs_accuracy_pct_std"),
                "precision_pct": read_float(best, "fkgs_precision_pct_mean"),
                "precision_std_pct": read_float(best, "fkgs_precision_pct_std"),
                "recall_pct": read_float(best, "fkgs_recall_pct_mean"),
                "recall_std_pct": read_float(best, "fkgs_recall_pct_std"),
                "specificity_pct": None,
                "specificity_std_pct": None,
                "f1_pct": None,
                "f1_std_pct": None,
                "auc_pct": None,
                "auc_std_pct": None,
                "train_time_s": read_float(best, "fkgs_full_train_time_seconds_mean"),
                "train_time_std_s": read_float(best, "fkgs_full_train_time_seconds_std"),
                "test_time_s": read_float(best, "fkgs_test_time_seconds_mean"),
                "test_time_std_s": read_float(best, "fkgs_test_time_seconds_std"),
                "total_time_s": read_float(best, "fkgs_end_to_end_time_seconds_mean"),
                "total_time_std_s": read_float(best, "fkgs_end_to_end_time_seconds_std"),
                "source_path": str(fkgs_summary),
            }
        )


def add_fkg_rows(
    rows: list[dict],
    fkg_summary: Path,
    fkg_summary_rows: list[dict[str, str]],
    protocol: str,
) -> None:
    for modality in ("image", "table", "fusion"):
        candidates = [row for row in fkg_summary_rows if row.get("modality") == modality]
        if not candidates:
            raise RuntimeError(f"Missing native FKG summary for modality: {modality}")
        best = candidates[0]
        label, data_type = FKGS_LABELS[modality]
        train_time = read_float(best, "fkg_full_train_time_seconds_mean")
        train_time_std = read_float(best, "fkg_full_train_time_seconds_std")
        if train_time is None:
            train_time = read_float(best, "fkg_train_time_seconds_mean")
            train_time_std = read_float(best, "fkg_train_time_seconds_std")
        rows.append(
            {
                "model": label,
                "data_type": data_type,
                "source_family": "FKG_KFold",
                "eval_split": "val_mean_5fold",
                "protocol": protocol,
                "selected_config": (
                    f"native FKG rerun; folds={best['folds']}; "
                    f"features={float(best['feature_count_mean']):.0f}"
                ),
                "accuracy_pct": read_percent_fraction(best, "fkg_accuracy_mean"),
                "accuracy_std_pct": read_percent_fraction(best, "fkg_accuracy_std"),
                "precision_pct": read_percent_fraction(best, "fkg_precision_mean"),
                "precision_std_pct": read_percent_fraction(best, "fkg_precision_std"),
                "recall_pct": read_percent_fraction(best, "fkg_recall_mean"),
                "recall_std_pct": read_percent_fraction(best, "fkg_recall_std"),
                "specificity_pct": read_percent_fraction(best, "fkg_specificity_mean"),
                "specificity_std_pct": read_percent_fraction(best, "fkg_specificity_std"),
                "f1_pct": read_percent_fraction(best, "fkg_f1_mean"),
                "f1_std_pct": read_percent_fraction(best, "fkg_f1_std"),
                "auc_pct": read_percent_fraction(best, "fkg_auc_mean"),
                "auc_std_pct": read_percent_fraction(best, "fkg_auc_std"),
                "train_time_s": train_time,
                "train_time_std_s": train_time_std,
                "test_time_s": read_float(best, "fkg_test_time_seconds_mean"),
                "test_time_std_s": read_float(best, "fkg_test_time_seconds_std"),
                "total_time_s": read_float(best, "fkg_end_to_end_time_seconds_mean"),
                "total_time_std_s": read_float(best, "fkg_end_to_end_time_seconds_std"),
                "source_path": str(fkg_summary),
            }
        )


def write_outputs(
    rows: list[dict],
    output_stem: Path,
    deep_summaries: list[Path],
    fkgs_summary: Path,
    fkgs_tables: Path | None,
    fkg_summary: Path | None,
    deep_sources: list[dict],
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    md_path = output_stem.with_suffix(".md")

    fieldnames = [
        "model",
        "data_type",
        "source_family",
        "eval_split",
        "protocol",
        "selected_config",
        "accuracy_pct",
        "accuracy_std_pct",
        "precision_pct",
        "precision_std_pct",
        "recall_pct",
        "recall_std_pct",
        "specificity_pct",
        "specificity_std_pct",
        "f1_pct",
        "f1_std_pct",
        "auc_pct",
        "auc_std_pct",
        "train_time_s",
        "train_time_std_s",
        "test_time_s",
        "test_time_std_s",
        "total_time_s",
        "total_time_std_s",
        "source_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})

    md_lines = [
        "# Diabetic Retinopathy Model Comparison - KFold Rerun",
        "",
        "Ngu\u1ed3n deep baseline: "
        + ", ".join(f"`{relative_to_project(path)}`" for path in deep_summaries)
        + ".",
        f"Ngu\u1ed3n FKGS: `{relative_to_project(fkgs_summary)}`.",
        (
            f"Ngu\u1ed3n native FKG: `{relative_to_project(fkg_summary)}`."
            if fkg_summary
            else "Ngu\u1ed3n native FKG: kh\u00f4ng cung c\u1ea5p, d\u00f9ng FKGS summary."
        ),
        "Giao th\u1ee9c: patient-aware 5-fold validation, kh\u00f4ng d\u00f9ng outer test trong l\u1ea7n t\u1ed5ng h\u1ee3p n\u00e0y.",
        "",
        deep_backbone_note(rows),
        "",
        "| M\u00f4 h\u00ecnh | Ki\u1ec3u d\u1eef li\u1ec7u | Protocol | Acc (%) | Precision (%) | Recall/Sensitivity (%) | Specificity (%) | F1 (%) | AUC (%) | Train (s) | Test (s) | Total (s) | Ghi ch\u00fa |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {model} | {data_type} | {protocol} | {accuracy} | {precision} | {recall} | {specificity} | "
            "{f1} | {auc} | {train} | {test} | {total} | {note} |".format(
                model=row["model"],
                data_type=row["data_type"],
                protocol=row["protocol"],
                accuracy=fmt_pm(row["accuracy_pct"], row["accuracy_std_pct"]),
                precision=fmt_pm(row["precision_pct"], row["precision_std_pct"]),
                recall=fmt_pm(row["recall_pct"], row["recall_std_pct"]),
                specificity=fmt_pm(row["specificity_pct"], row["specificity_std_pct"]),
                f1=fmt_pm(row["f1_pct"], row["f1_std_pct"]),
                auc=fmt_pm(row["auc_pct"], row["auc_std_pct"]),
                train=fmt_pm(row["train_time_s"], row["train_time_std_s"], 2),
                test=fmt_pm(row["test_time_s"], row["test_time_std_s"], 2),
                total=fmt_pm(row["total_time_s"], row["total_time_std_s"], 2),
                note=row["selected_config"],
            )
        )

    if fkgs_tables:
        md_lines += [
            "",
            "## FKGS all ran/e tables",
            "",
            f"B\u1ea3ng \u0111\u1ea7y \u0111\u1ee7 theo `ran` v\u00e0 `epsilon`: `{relative_to_project(fkgs_tables)}`.",
        ]
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[DONE] Comparison CSV: {csv_path}")
    print(f"[DONE] Comparison Markdown: {md_path}")


def main() -> int:
    args = parse_args()
    deep_summaries = [resolve_path(path) for path in args.deep_summary]
    fkgs_summary = resolve_path(args.fkgs_summary)
    fkgs_tables = resolve_path(args.fkgs_tables) if args.fkgs_tables else None
    fkg_summary = resolve_path(args.fkg_summary) if args.fkg_summary else None
    output_stem = resolve_path(args.output_stem)

    for deep_summary in deep_summaries:
        if not deep_summary.exists():
            raise FileNotFoundError(deep_summary)
    if not fkgs_summary.exists():
        raise FileNotFoundError(fkgs_summary)
    if fkgs_tables and not fkgs_tables.exists():
        raise FileNotFoundError(fkgs_tables)
    if fkg_summary and not fkg_summary.exists():
        raise FileNotFoundError(fkg_summary)

    deep_config = resolve_path(args.deep_config) if args.deep_config else None
    deep_sources = []
    for deep_summary in deep_summaries:
        config = read_config(
            deep_summary,
            deep_config if deep_config and len(deep_summaries) == 1 else None,
        )
        summary_rows = read_csv(deep_summary)
        deep_sources.append(
            {
                "path": deep_summary,
                "rows": summary_rows,
                "config": config,
                "by_model": {
                    row["model"]: row
                    for row in summary_rows
                    if row.get("eval_split") == "val"
                },
            }
        )
    fkgs_rows = read_csv(fkgs_summary)
    fkg_rows = read_csv(fkg_summary) if fkg_summary else []

    rows: list[dict] = []
    add_deep_rows(rows, deep_sources, args.protocol)
    if fkg_summary:
        add_fkg_rows(rows, fkg_summary, fkg_rows, args.protocol)
    else:
        add_fkgs_rows(rows, fkgs_summary, fkgs_rows, args.protocol)
    write_outputs(rows, output_stem, deep_summaries, fkgs_summary, fkgs_tables, fkg_summary, deep_sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
