import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
workspace_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from module.FKG.FKG_S import FKGS


MODALITIES = [
    {
        "key": "table_full",
        "model": "FKG-UM",
        "label": "Dữ liệu dạng bảng full",
        "source_modality": "Diabetic Retinopathy Metadata Feature",
        "run_modality_prefix": "Diabetic Retinopathy Metadata Feature Full Table Rerun",
    },
    {
        "key": "image_full",
        "model": "FKG-UM",
        "label": "Dữ liệu ảnh",
        "source_modality": "Diabetic Retinopathy Image Feature",
        "run_modality_prefix": "Diabetic Retinopathy Image Feature Full Attributes Rerun",
    },
    {
        "key": "fusion_ft_selection",
        "model": "FKG-MM",
        "label": "Dữ liệu ảnh+bảng",
        "source_modality": "Diabetic Retinopathy Feature FT Selection",
        "run_modality_prefix": "Diabetic Retinopathy Feature FT Selection Rerun",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FKGS for table-full, image, and fusion diabetic retinopathy tables."
    )
    parser.add_argument("--ran", type=int, nargs="+", default=[15, 20])
    parser.add_argument("--e", type=float, nargs="+", default=[0.2, 0.3])
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--run-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", type=Path, default=workspace_root / "result")
    return parser.parse_args()


def load_fis_rules(source_modality):
    frb_dir = project_root / "data" / "FIS" / "output" / source_modality / "FRB"
    train_path = frb_dir / "TrainDataRule.csv"
    test_path = frb_dir / "TestDataRule.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train rules: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test rules: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    base = [[int(float(x)) for x in row] for row in train_df.values]
    test = [[int(float(x)) for x in row] for row in test_df.values]
    return pd.DataFrame(base), test, train_df.shape, test_df.shape, train_path, test_path


def config_seed(base_seed, modality_index, ran, e_value):
    return int(base_seed + modality_index * 100000 + ran * 1000 + round(e_value * 1000))


def round2(value):
    return f"{float(value):.2f}"


def write_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "table_no",
        "model",
        "modality",
        "modality_key",
        "source_modality",
        "run_modality",
        "ran_pct",
        "epsilon",
        "accuracy_pct",
        "accuracy_std",
        "train_time_s",
        "train_time_std",
        "test_time_s",
        "test_time_std",
        "total_time_s",
        "total_time_std",
        "sampling_time_s",
        "sampling_time_std",
        "train_rule_shape",
        "test_rule_shape",
        "train_rule_path",
        "test_rule_path",
        "seed",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, md_path, csv_path, run_date):
    title_by_config = {
        (15, 0.2): "Bảng 3.2. Phương pháp lựa chọn thuộc tính với tỉ lệ mẫu 15% và ngưỡng sai số 0.2",
        (15, 0.3): "Bảng 3.3. Phương pháp lựa chọn thuộc tính với tỉ lệ mẫu 15% và ngưỡng sai số 0.3",
        (20, 0.2): "Bảng 3.4. Phương pháp lựa chọn thuộc tính với tỉ lệ mẫu 20% và ngưỡng sai số 0.2",
        (20, 0.3): "Bảng 3.5. Phương pháp lựa chọn thuộc tính với tỉ lệ mẫu 20% và ngưỡng sai số 0.3",
    }
    lines = [
        f"# Kết quả FKGS bảng full, ảnh và fusion - {run_date}",
        "",
        f"CSV chi tiết: `{csv_path}`",
        "",
        "Ghi chú: các giá trị trong bảng là trung bình 5 lượt chạy; độ lệch chuẩn lưu trong CSV chi tiết.",
        "",
    ]
    for (ran, e_value), title in title_by_config.items():
        lines.extend(
            [
                f"## {title}",
                "",
                "| Mô hình | Mô thức | Acc (%) | Thời gian huấn luyện | Thời gian kiểm tra | Tổng thời gian (s) |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        subset = [r for r in rows if int(r["ran_pct"]) == ran and abs(float(r["epsilon"]) - e_value) < 1e-9]
        subset.sort(key=lambda r: ["table_full", "image_full", "fusion_ft_selection"].index(r["modality_key"]))
        for row in subset:
            lines.append(
                "| {model} | {modality} | {acc} | {train} | {test} | {total} |".format(
                    model=row["model"],
                    modality=row["modality"],
                    acc=round2(row["accuracy_pct"]),
                    train=round2(row["train_time_s"]),
                    test=round2(row["test_time_s"]),
                    total=round2(row["total_time_s"]),
                )
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(project_root)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.run_date
    csv_path = out_dir / f"diabetic_retinopathy_fkgs_table_image_fusion_summary_{stamp}.csv"
    md_path = out_dir / f"diabetic_retinopathy_fkgs_table_image_fusion_tables_{stamp}.md"
    progress_path = out_dir / f"diabetic_retinopathy_fkgs_table_image_fusion_progress_{stamp}.json"

    rows = []
    table_no_by_config = {(15, 0.2): "3.2", (15, 0.3): "3.3", (20, 0.2): "3.4", (20, 0.3): "3.5"}

    print("=" * 100)
    print("FKGS table-full + image + fusion rerun")
    print("Run date:", stamp)
    print("Output CSV:", csv_path)
    print("Output Markdown:", md_path)
    print("=" * 100)

    for modality_index, modality in enumerate(MODALITIES):
        base, test, train_shape, test_shape, train_path, test_path = load_fis_rules(modality["source_modality"])
        print()
        print(f"[MODALITY] {modality['label']}")
        print(f"Source modality: {modality['source_modality']}")
        print(f"TrainDataRule: {train_shape} - {train_path}")
        print(f"TestDataRule : {test_shape} - {test_path}")

        for ran in args.ran:
            for e_value in args.e:
                seed = config_seed(args.seed, modality_index, ran, e_value)
                random.seed(seed)
                np.random.seed(seed)
                run_modality = f"{modality['run_modality_prefix']} {stamp}"
                print()
                print(f"[RUN] modality={modality['key']} ran={ran} e={e_value} seed={seed}")
                fkg_instance = FKGS()
                summary = fkg_instance.FKGS(
                    df=base,
                    testdf=test,
                    Turn=None,
                    Modality=run_modality,
                    ran=ran,
                    e=e_value,
                    folderPath=project_root,
                )
                row = {
                    "table_no": table_no_by_config.get((int(ran), float(e_value)), ""),
                    "model": modality["model"],
                    "modality": modality["label"],
                    "modality_key": modality["key"],
                    "source_modality": modality["source_modality"],
                    "run_modality": run_modality,
                    "ran_pct": int(ran),
                    "epsilon": float(e_value),
                    "accuracy_pct": summary["accuracy_mean"],
                    "accuracy_std": summary["accuracy_std"],
                    "train_time_s": summary["train_time_mean"],
                    "train_time_std": summary["train_time_std"],
                    "test_time_s": summary["test_time_mean"],
                    "test_time_std": summary["test_time_std"],
                    "total_time_s": summary["total_time_mean"],
                    "total_time_std": summary["total_time_std"],
                    "sampling_time_s": summary["sampling_time_mean"],
                    "sampling_time_std": summary["sampling_time_std"],
                    "train_rule_shape": json.dumps(list(train_shape)),
                    "test_rule_shape": json.dumps(list(test_shape)),
                    "train_rule_path": str(train_path),
                    "test_rule_path": str(test_path),
                    "seed": seed,
                }
                rows.append(row)
                write_csv(rows, csv_path)
                write_markdown(rows, md_path, csv_path, stamp)
                progress_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                print(
                    "[DONE] {modality} ran={ran} e={e}: acc={acc:.2f}, train={train:.2f}s, "
                    "test={test_time:.2f}s, total={total:.2f}s".format(
                        modality=modality["label"],
                        ran=ran,
                        e=e_value,
                        acc=summary["accuracy_mean"],
                        train=summary["train_time_mean"],
                        test_time=summary["test_time_mean"],
                        total=summary["total_time_mean"],
                    )
                )

    write_csv(rows, csv_path)
    write_markdown(rows, md_path, csv_path, stamp)
    progress_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 100)
    print("[OK] Completed all configurations")
    print("[OK] CSV:", csv_path)
    print("[OK] Markdown:", md_path)
    print("[OK] Progress JSON:", progress_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
