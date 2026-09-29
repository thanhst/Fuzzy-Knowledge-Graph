import csv
import os


def package_available(package_root):
    return os.path.isfile(os.path.join(package_root, "frb_validation_summary.csv"))


def load_frb_folds(package_root, modality="table"):
    summary_path = os.path.join(package_root, "frb_validation_summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"FRB package summary not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["modality"] == modality]
    if not rows:
        raise ValueError(f"No modality '{modality}' in FRB package.")

    folds = []
    for row in sorted(rows, key=lambda item: int(item["fold"])):
        fold = int(row["fold"])
        column_map = _load_column_map(package_root, modality, fold,
                                      row["rule_list_relpath"])
        feature_modalities = _load_feature_modalities(
            package_root, modality, fold, column_map)
        train_path = os.path.join(package_root, *row["train_rule_relpath"].split("/"))
        test_path = os.path.join(package_root, *row["test_rule_relpath"].split("/"))
        train_rows = _read_rule_rows(train_path)
        test_rows = _read_rule_rows(test_path)

        label_column = next(index for index, item in column_map.items()
                            if item["role"].startswith("label"))
        labels = sorted({entry[label_column] for entry in train_rows + test_rows},
                        key=lambda value: float(value))
        label_map = {value: index for index, value in enumerate(labels)}
        train_manifest = _load_split_rows(package_root, fold, "train")
        validation_manifest = _load_split_rows(package_root, fold, "val")
        test_patient_ids = [item["patient_id"] for item in validation_manifest]
        if len(test_patient_ids) != len(test_rows):
            raise ValueError(
                f"Fold {fold}: {len(test_rows)} FRB test rows but "
                f"{len(test_patient_ids)} validation patient IDs."
            )
        if len(train_manifest) != int(row["train_source_rows"]):
            raise ValueError(f"Fold {fold}: source train row count differs from manifest.")
        if len(train_rows) != int(row["train_rule_rows"]):
            raise ValueError(f"Fold {fold}: FRB train row count differs from summary.")
        train_patient_ids = {item["patient_id"] for item in train_manifest}
        validation_patient_ids = set(test_patient_ids)
        actual_overlap = train_patient_ids & validation_patient_ids
        if actual_overlap:
            raise ValueError(f"Fold {fold}: {len(actual_overlap)} patient IDs overlap.")
        if len(train_patient_ids) != int(row["train_patient_count"]) or len(validation_patient_ids) != int(row["test_patient_count"]):
            raise ValueError(f"Fold {fold}: patient counts differ from summary.")
        for index, (rule, item) in enumerate(zip(test_rows, validation_manifest)):
            if label_map[rule[label_column]] != int(item["retinopathy"]):
                raise ValueError(f"Fold {fold}: validation label differs at row {index}.")

        train_records = [
            _to_record(entry, column_map, feature_modalities, label_column, label_map,
                       f"fold-{fold}-train-row-{index}")
            for index, entry in enumerate(train_rows)
        ]
        test_records = [
            _to_record(entry, column_map, feature_modalities, label_column, label_map,
                       patient_id)
            for entry, patient_id in zip(test_rows, test_patient_ids)
        ]
        train_patients = int(row["train_patient_count"])
        test_patients = int(row["test_patient_count"])
        overlap = len(actual_overlap)
        if overlap != 0:
            raise ValueError(f"Fold {fold} has patient overlap count {overlap}.")
        if int(row["patient_overlap_count"]) != overlap:
            raise ValueError(f"Fold {fold}: summary overlap differs from manifest.")

        folds.append({
            "fold": fold,
            "train_records": train_records,
            "test_records": test_records,
            "metadata": {
                "source": "brset_frb_package",
                "modality": modality,
                "display_name": row["display_name"],
                "fis_run_tag": row["fis_run_tag"],
                "smote": row["smote"],
                "feature_count": int(row["feature_count"]),
                "modalities": sorted(set(feature_modalities.values())),
                "input_graph_kind": (
                    "FKG-MM" if len(set(feature_modalities.values())) > 1 else "FKG-UM"),
                "train_patient_count": train_patients,
                "test_patient_count": test_patients,
                "patient_overlap_count": overlap,
                "manifest_image_id_match": row["manifest_image_id_match"].lower() == "true",
                "train_rule_path": train_path,
                "test_rule_path": test_path,
            },
        })
    return folds


def _load_column_map(package_root, modality, fold, rule_relpath):
    path = os.path.join(package_root, "rule_column_map.csv")
    result = {}
    normalized_rule_path = rule_relpath.replace("\\", "/")
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["modality"] == modality and int(row["fold"]) == fold
                    and row["rule_file_relpath"].replace("\\", "/") == normalized_rule_path):
                result[row["rule_csv_header"]] = row
    if not result:
        raise ValueError(f"Missing column map for modality={modality}, fold={fold}.")
    return result


def _load_feature_modalities(package_root, modality, fold, column_map):
    if modality in {"image", "table"}:
        return {
            column: modality for column, item in column_map.items()
            if item["role"].startswith("feature")
        }

    path = os.path.join(package_root, "rule_column_map.csv")
    base_sources = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (int(row["fold"]) == fold and row["modality"] in {"image", "table"}
                    and row["role"].startswith("feature")):
                base_sources.setdefault(row["column_name"], set()).add(row["modality"])

    result = {}
    for column, item in column_map.items():
        if not item["role"].startswith("feature"):
            continue
        sources = base_sources.get(item["column_name"], set())
        result[column] = next(iter(sources)) if len(sources) == 1 else "fusion"

    if modality == "fusion" and set(result.values()) != {"image", "table"}:
        raise ValueError(
            f"Fold {fold}: fusion input must resolve to image and table features; "
            f"found {sorted(set(result.values()))}."
        )
    return result


def _read_rule_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_split_rows(package_root, fold, split):
    path = os.path.join(package_root, "root_split", "train_kfold",
                        f"fold_{fold}", f"{split}.csv")
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _to_record(row, column_map, feature_modalities, label_column, label_map, patient_id):
    tokens = []
    for column, item in sorted(column_map.items(), key=lambda pair: int(pair[0])):
        if column == label_column:
            continue
        raw_value = row[column]
        numeric = float(raw_value)
        level = str(int(numeric)) if numeric.is_integer() else raw_value
        source_modality = feature_modalities[column]
        tokens.append(f"{source_modality}::{item['column_name']}=L{level}")
    return {
        "antecedent_tokens": tokens,
        "label": label_map[row[label_column]],
        "patient_id": str(patient_id),
    }
