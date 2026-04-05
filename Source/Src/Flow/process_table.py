#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


CODE_PATTERN = re.compile(r"^\d{2}\.\d{4}$")


@dataclass
class PatientRecord:
    patient_id: str
    name: str
    age: float
    class_name: str
    class_level: float
    gender: str
    gender_code: float
    lesion_count: int
    early_count: int
    moderate_count: int
    severe_count: int
    mean_icdas: float
    std_icdas: float
    max_icdas: int
    white_opaque_count: int
    light_brown_count: int
    smooth_surface_count: int
    rough_surface_count: int
    yellow_brown_count: int
    black_dark_count: int
    lesion_size_mean: float
    soft_base_count: int
    hard_base_count: int
    stimulation_yes_count: int
    stimulation_no_count: int
    outcome: int


def source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_status_cell(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if isinstance(value, pd.Timestamp):
        # Excel can auto-convert "1.2" / "2.2" to date with "d.m" format.
        return f"{int(value.day)}.{int(value.month)}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text if text else None

    text = str(value).strip().replace("\n", "").replace(" ", "")
    return text or None


def parse_status_codes(value: object) -> List[int]:
    text = normalize_status_cell(value)
    if not text:
        return []

    cleaned = re.sub(r"[^0-9.]", "", text)
    if not cleaned:
        return []

    codes: List[int] = []
    for token in cleaned.split("."):
        if not token:
            continue
        try:
            codes.append(int(token))
        except ValueError:
            continue
    return codes


def parse_icdas(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_class_level(class_name: object) -> float:
    if class_name is None or (isinstance(class_name, float) and np.isnan(class_name)):
        return 0.0
    text = str(class_name).strip()
    match = re.search(r"\d+", text)
    if not match:
        return 0.0
    return float(match.group())


def parse_gender_code(gender: object) -> float:
    if gender is None or (isinstance(gender, float) and np.isnan(gender)):
        return 0.0
    text = str(gender).strip().lower()
    if text.startswith("nam"):
        return 1.0
    if text.startswith("nữ") or text.startswith("nu"):
        return 0.0
    return 0.0


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_patient_record(
    patient_row: pd.Series,
    icdas_row: pd.Series,
    tooth_col_indices: Iterable[int],
) -> PatientRecord:
    patient_id = str(patient_row.iloc[4]).strip()
    name = "" if pd.isna(patient_row.iloc[0]) else str(patient_row.iloc[0]).strip()
    class_name = "" if pd.isna(patient_row.iloc[2]) else str(patient_row.iloc[2]).strip()
    gender = "" if pd.isna(patient_row.iloc[3]) else str(patient_row.iloc[3]).strip()

    age = to_float(patient_row.iloc[1], 0.0)
    class_level = parse_class_level(class_name)
    gender_code = parse_gender_code(gender)

    icdas_values: List[int] = []
    white_opaque_count = 0
    light_brown_count = 0
    smooth_surface_count = 0
    rough_surface_count = 0
    yellow_brown_count = 0
    black_dark_count = 0
    lesion_size_values: List[int] = []
    soft_base_count = 0
    hard_base_count = 0
    stimulation_yes_count = 0
    stimulation_no_count = 0

    for col_idx in tooth_col_indices:
        icdas = parse_icdas(icdas_row.iloc[col_idx])
        if icdas is None:
            continue

        status_codes = parse_status_codes(patient_row.iloc[col_idx])
        icdas_values.append(icdas)

        if icdas <= 2:
            if len(status_codes) >= 1:
                if status_codes[0] == 1:
                    white_opaque_count += 1
                elif status_codes[0] == 2:
                    light_brown_count += 1
            if len(status_codes) >= 2:
                if status_codes[1] == 1:
                    smooth_surface_count += 1
                elif status_codes[1] == 2:
                    rough_surface_count += 1
        else:
            if len(status_codes) >= 1:
                if status_codes[0] == 1:
                    yellow_brown_count += 1
                elif status_codes[0] == 2:
                    black_dark_count += 1
            if len(status_codes) >= 2 and status_codes[1] in {1, 2, 3, 4}:
                lesion_size_values.append(status_codes[1])
            if len(status_codes) >= 3:
                if status_codes[2] == 1:
                    soft_base_count += 1
                elif status_codes[2] == 2:
                    hard_base_count += 1
            if len(status_codes) >= 4:
                if status_codes[3] == 1:
                    stimulation_yes_count += 1
                elif status_codes[3] == 2:
                    stimulation_no_count += 1

    lesion_count = len(icdas_values)
    early_count = int(sum(1 for x in icdas_values if 1 <= x <= 2))
    moderate_count = int(sum(1 for x in icdas_values if 3 <= x <= 4))
    severe_count = int(sum(1 for x in icdas_values if x >= 5))
    mean_icdas = float(np.mean(icdas_values)) if icdas_values else 0.0
    std_icdas = float(np.std(icdas_values)) if icdas_values else 0.0
    max_icdas = int(max(icdas_values)) if icdas_values else 0
    lesion_size_mean = float(np.mean(lesion_size_values)) if lesion_size_values else 0.0

    # 4-level label aligned with DLFKG document:
    # 0=Normal, 1=Mild, 2=Moderate, 3=Severe
    if lesion_count == 0:
        outcome = 0
    elif max_icdas <= 2:
        outcome = 1
    elif max_icdas <= 4:
        outcome = 2
    else:
        outcome = 3

    return PatientRecord(
        patient_id=patient_id,
        name=name,
        age=age,
        class_name=class_name,
        class_level=class_level,
        gender=gender,
        gender_code=gender_code,
        lesion_count=lesion_count,
        early_count=early_count,
        moderate_count=moderate_count,
        severe_count=severe_count,
        mean_icdas=mean_icdas,
        std_icdas=std_icdas,
        max_icdas=max_icdas,
        white_opaque_count=white_opaque_count,
        light_brown_count=light_brown_count,
        smooth_surface_count=smooth_surface_count,
        rough_surface_count=rough_surface_count,
        yellow_brown_count=yellow_brown_count,
        black_dark_count=black_dark_count,
        lesion_size_mean=lesion_size_mean,
        soft_base_count=soft_base_count,
        hard_base_count=hard_base_count,
        stimulation_yes_count=stimulation_yes_count,
        stimulation_no_count=stimulation_no_count,
        outcome=outcome,
    )


def records_to_frames(records: List[PatientRecord]) -> Dict[str, pd.DataFrame]:
    decoded_rows = []
    icta_rows = []

    for r in records:
        decoded_rows.append(
            {
                "patient_id": r.patient_id,
                "name": r.name,
                "age": r.age,
                "class_name": r.class_name,
                "gender": r.gender,
                "class_level": r.class_level,
                "gender_code": r.gender_code,
                "lesion_count": r.lesion_count,
                "early_count": r.early_count,
                "moderate_count": r.moderate_count,
                "severe_count": r.severe_count,
                "mean_icdas": r.mean_icdas,
                "std_icdas": r.std_icdas,
                "max_icdas": r.max_icdas,
                "white_opaque_count": r.white_opaque_count,
                "light_brown_count": r.light_brown_count,
                "smooth_surface_count": r.smooth_surface_count,
                "rough_surface_count": r.rough_surface_count,
                "yellow_brown_count": r.yellow_brown_count,
                "black_dark_count": r.black_dark_count,
                "lesion_size_mean": r.lesion_size_mean,
                "soft_base_count": r.soft_base_count,
                "hard_base_count": r.hard_base_count,
                "stimulation_yes_count": r.stimulation_yes_count,
                "stimulation_no_count": r.stimulation_no_count,
                "Outcome": r.outcome,
            }
        )

        icta_rows.append(
            {
                "patient_id": r.patient_id,
                "Age": r.age,
                "ClassLevel": r.class_level,
                "GenderCode": r.gender_code,
                "LesionCount": r.lesion_count,
                "EarlyCount": r.early_count,
                "ModerateCount": r.moderate_count,
                "SevereCount": r.severe_count,
                "MeanICDAS": r.mean_icdas,
                "StdICDAS": r.std_icdas,
                "MaxICDAS": r.max_icdas,
                "RoughSurfaceCount": r.rough_surface_count,
                "SoftBaseCount": r.soft_base_count,
                "StimulusYesCount": r.stimulation_yes_count,
                "Outcome": r.outcome,
            }
        )

    decoded_df = pd.DataFrame(decoded_rows).sort_values("patient_id").reset_index(drop=True)
    icta_df = pd.DataFrame(icta_rows).sort_values("patient_id").reset_index(drop=True)
    return {"decoded": decoded_df, "icta": icta_df}


def process_table(excel_path: Path, output_dir: Path) -> Dict[str, Path]:
    raw = pd.read_excel(excel_path, header=None)

    records: List[PatientRecord] = []
    tooth_col_indices = range(6, 34)

    for idx in range(len(raw)):
        row = raw.iloc[idx]
        code = row.iloc[4] if row.shape[0] > 4 else None
        if not isinstance(code, str) or not CODE_PATTERN.match(code.strip()):
            continue
        if idx + 1 >= len(raw):
            continue

        record = build_patient_record(
            patient_row=row,
            icdas_row=raw.iloc[idx + 1],
            tooth_col_indices=tooth_col_indices,
        )
        records.append(record)

    frames = records_to_frames(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    decoded_csv = output_dir / "patient_table_decoded.csv"
    icta_csv = output_dir / "ICTA_table.csv"
    frames["decoded"].to_csv(decoded_csv, index=False, encoding="utf-8")
    frames["icta"].to_csv(icta_csv, index=False, encoding="utf-8")

    appendix_mapping = {
        "icdas_1_2": {
            "color": {"1": "Trang duc", "2": "Nau nhat"},
            "surface": {"1": "Tron nhan", "2": "Tho rap/xop/nham"},
        },
        "icdas_3_plus": {
            "color": {"1": "Vang-nau", "2": "Den/nau den"},
            "size": {"1": "Nho", "2": "Vua", "3": "Lon", "4": "Rat lon"},
            "base": {"1": "Mem/mun", "2": "Cung"},
            "stimulation": {"1": "Co", "2": "Khong"},
        },
        "outcome_map": {
            "0": "Binh thuong",
            "1": "Nhe (ICDAS <= 2)",
            "2": "Trung binh (ICDAS 3-4)",
            "3": "Nang (ICDAS >= 5)",
        },
    }
    mapping_path = output_dir / "appendix_code_mapping.json"
    mapping_path.write_text(json.dumps(appendix_mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"decoded_csv": decoded_csv, "icta_csv": icta_csv, "mapping_json": mapping_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Process coded dental excel into ICTA-like table.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=source_root() / "Data" / "Raw data" / "code excel.xlsx",
        help="Path to code excel file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=source_root() / "Data" / "processing" / "table",
        help="Output directory for processed table files.",
    )
    args = parser.parse_args()

    if not args.excel.exists():
        raise FileNotFoundError(f"Excel file not found: {args.excel}")

    outputs = process_table(args.excel, args.out_dir)
    print(f"[OK] Decoded table: {outputs['decoded_csv']}")
    print(f"[OK] ICTA-like table: {outputs['icta_csv']}")
    print(f"[OK] Mapping file: {outputs['mapping_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
