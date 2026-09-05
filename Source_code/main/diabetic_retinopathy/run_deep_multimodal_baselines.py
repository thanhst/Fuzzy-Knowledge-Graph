from __future__ import annotations

import argparse
import atexit
import copy
import csv
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    class _MissingNN:
        class Module:
            pass

    torch = None
    nn = _MissingNN()
    Dataset = object
    DataLoader = None

try:
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:
    transforms = None
    ResNet18_Weights = None
    resnet18 = None


LABEL_COLUMN = "retinopathy"
SOURCE_LABEL_COLUMNS = {"retinopathy", "diabetic_retinopathy"}
DEFAULT_SPLIT_ROOT = Path("ROOT_DATA/train_test_selection")
DEFAULT_TABULAR_CSV = Path("Source_code/data/Dataset_diabetic/data_process.csv")
DEFAULT_MODELS = ("mlp", "resnet", "early_fusion", "late_fusion")
DEFAULT_TABULAR_COLUMNS = (
    "patient_age",
    "diabetes_time_y",
    "insuline",
    "patient_sex",
    "exam_eye",
    "diabetes",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "image_field",
    "quality",
)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def enable_console_logging(log_path: Path) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def restore_streams() -> None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(restore_streams)
    print(f"[LOG] Console log: {log_path}")
    return log_path


@dataclass
class TabularPreprocessor:
    feature_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    numeric_means: pd.Series
    numeric_stds: pd.Series
    categorical_dummy_columns: List[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_columns: Sequence[str]) -> "TabularPreprocessor":
        numeric_columns: List[str] = []
        categorical_columns: List[str] = []
        for column in feature_columns:
            values = frame[column]
            numeric_values = pd.to_numeric(values, errors="coerce")
            non_null_count = int(values.notna().sum())
            if non_null_count and int(numeric_values.notna().sum()) == non_null_count:
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)

        if numeric_columns:
            numeric_frame = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
            numeric_means = numeric_frame.mean().fillna(0.0)
            numeric_stds = numeric_frame.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        else:
            numeric_means = pd.Series(dtype=float)
            numeric_stds = pd.Series(dtype=float)

        if categorical_columns:
            categorical_frame = frame[categorical_columns].astype("string").fillna("__missing__")
            categorical_dummies = pd.get_dummies(categorical_frame, dummy_na=False, dtype=float)
            categorical_dummy_columns = list(categorical_dummies.columns)
        else:
            categorical_dummy_columns = []

        return cls(
            feature_columns=list(feature_columns),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            numeric_means=numeric_means,
            numeric_stds=numeric_stds,
            categorical_dummy_columns=categorical_dummy_columns,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: List[pd.DataFrame] = []
        if self.numeric_columns:
            numeric_frame = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
            numeric_frame = numeric_frame.fillna(self.numeric_means)
            numeric_frame = (numeric_frame - self.numeric_means) / self.numeric_stds
            parts.append(numeric_frame.astype(float))

        if self.categorical_columns:
            categorical_frame = frame[self.categorical_columns].astype("string").fillna("__missing__")
            categorical_dummies = pd.get_dummies(categorical_frame, dummy_na=False, dtype=float)
            categorical_dummies = categorical_dummies.reindex(
                columns=self.categorical_dummy_columns,
                fill_value=0.0,
            )
            parts.append(categorical_dummies.astype(float))

        if not parts:
            return np.zeros((len(frame), 0), dtype=np.float32)
        return pd.concat(parts, axis=1).to_numpy(dtype=np.float32)


class RetinopathyDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        label_to_index: Dict[int, int],
        image_transform=None,
        tabular_matrix: np.ndarray | None = None,
        use_images: bool = False,
        use_tabular: bool = False,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.image_transform = image_transform
        self.tabular_matrix = tabular_matrix
        self.use_images = use_images
        self.use_tabular = use_tabular
        if self.use_tabular and self.tabular_matrix is None:
            raise ValueError("Tabular matrix is required for this model")
        if self.use_tabular and len(self.tabular_matrix) != len(self.manifest):
            raise ValueError("Tabular matrix row count does not match manifest")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.manifest.iloc[index]
        label = int(row[LABEL_COLUMN])
        item: Dict[str, object] = {
            "label": torch.tensor(self.label_to_index[label], dtype=torch.long),
            "original_label": label,
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
        }
        if self.use_images:
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB")
                if self.image_transform is not None:
                    image = self.image_transform(image)
            item["image"] = image
        if self.use_tabular:
            item["tabular"] = torch.from_numpy(self.tabular_matrix[index])
        return item


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, image=None, tabular=None):
        return self.net(tabular)


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool, freeze_backbone: bool):
        super().__init__()
        if resnet18 is None:
            raise RuntimeError("torchvision is required for ResNet/image/fusion models")
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(pretrained=pretrained)
        self.output_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        if freeze_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        self.backbone = backbone

    def forward(self, image):
        return self.backbone(image)


class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool, freeze_backbone: bool, dropout: float):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.output_dim, num_classes),
        )

    def forward(self, image=None, tabular=None):
        return self.classifier(self.encoder(image))


class EarlyFusionClassifier(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        num_classes: int,
        pretrained: bool,
        freeze_backbone: bool,
        tabular_hidden_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.image_encoder = ResNetEncoder(pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden_dim),
            nn.LayerNorm(tabular_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim + tabular_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, image=None, tabular=None):
        image_features = self.image_encoder(image)
        tabular_features = self.tabular_encoder(tabular)
        return self.classifier(torch.cat([image_features, tabular_features], dim=1))


class LateFusionClassifier(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        num_classes: int,
        pretrained: bool,
        freeze_backbone: bool,
        tabular_hidden_dims: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        self.image_model = ImageClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
        self.tabular_model = MLPClassifier(
            input_dim=tabular_dim,
            num_classes=num_classes,
            hidden_dims=tabular_hidden_dims,
            dropout=dropout,
        )
        self.fusion_logits = nn.Parameter(torch.zeros(2))

    def forward(self, image=None, tabular=None):
        weights = torch.softmax(self.fusion_logits, dim=0)
        image_logits = self.image_model(image=image)
        tabular_logits = self.tabular_model(tabular=tabular)
        return weights[0] * image_logits + weights[1] * tabular_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate diabetic retinopathy deep baselines on patient-level "
            "train KFold splits: MLP, ResNet, Early Fusion, and Late Fusion."
        )
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--tabular-csv", type=Path, default=DEFAULT_TABULAR_CSV)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", *DEFAULT_MODELS],
    )
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pretrained-resnet", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--select-metric", default="auc", choices=["auc", "f1", "accuracy"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="After KFold, train each model on the full outer train split and evaluate outer test.",
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Validate manifests, patient splits, image paths, and tabular joins without importing PyTorch.",
    )
    parser.add_argument(
        "--console-log",
        type=Path,
        default=None,
        help="Path for saving the console output. Defaults to <results-dir>/console.log.",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable saving console output to a log file.",
    )
    parser.add_argument(
        "--tabular-columns",
        nargs="*",
        default=None,
        help="Override tabular feature columns. Defaults to selected train/test columns.",
    )
    return parser.parse_args()


def normalize_models(models: Sequence[str]) -> List[str]:
    if "all" in models:
        return list(DEFAULT_MODELS)
    return list(dict.fromkeys(models))


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def require_dependencies(models: Sequence[str]) -> None:
    if torch is None:
        raise SystemExit(
            "Missing dependency: torch. Install PyTorch in this environment before training."
        )
    if any(model in {"resnet", "early_fusion", "late_fusion"} for model in models):
        if transforms is None or resnet18 is None:
            raise SystemExit(
                "Missing dependency: torchvision. Install torchvision for ResNet/fusion models."
            )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str):
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"image_id": str, "patient_id": str})
    required = {"image_id", "patient_id", "image_path", LABEL_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame[LABEL_COLUMN] = pd.to_numeric(frame[LABEL_COLUMN], errors="raise").astype(int)
    missing_files = frame[~frame["image_path"].map(lambda value: Path(value).exists())]
    if not missing_files.empty:
        examples = ", ".join(missing_files["image_id"].head(5).astype(str))
        raise FileNotFoundError(f"{path} references missing image files: {examples}")
    return frame


def find_fold_dirs(split_root: Path, requested_folds: Sequence[int] | None) -> List[Path]:
    fold_root = split_root / "train_kfold"
    if not fold_root.exists():
        raise FileNotFoundError(
            f"{fold_root} does not exist. Run create_root_data_image_train_test_split.py first."
        )
    fold_dirs = sorted(
        [path for path in fold_root.glob("fold_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if requested_folds:
        requested = set(requested_folds)
        fold_dirs = [path for path in fold_dirs if int(path.name.split("_")[-1]) in requested]
    if not fold_dirs:
        raise FileNotFoundError(f"No matching fold directories found in {fold_root}")
    return fold_dirs


def load_tabular_source(tabular_csv: Path, feature_columns: Sequence[str] | None) -> pd.DataFrame:
    frame = pd.read_csv(tabular_csv, dtype={"image_id": str, "patient_id": str})
    if "image_id" not in frame.columns:
        raise ValueError(f"{tabular_csv} must contain image_id so it can join image folds safely")

    if feature_columns is None:
        missing_defaults = [column for column in DEFAULT_TABULAR_COLUMNS if column not in frame.columns]
        if missing_defaults:
            raise ValueError(
                f"{tabular_csv} is missing default tabular columns: {missing_defaults}. "
                "Pass --tabular-columns to override."
            )
        feature_columns = DEFAULT_TABULAR_COLUMNS
    else:
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{tabular_csv} is missing requested tabular columns: {missing}")

    keep_columns = ["image_id", *feature_columns]
    duplicates = frame[frame.duplicated("image_id", keep=False)]
    if not duplicates.empty:
        examples = ", ".join(duplicates["image_id"].head(5).astype(str))
        raise ValueError(f"Duplicate image_id values in {tabular_csv}: {examples}")
    return frame[keep_columns].copy()


def merge_tabular(manifest: pd.DataFrame, tabular_source: pd.DataFrame) -> pd.DataFrame:
    merged = manifest[["image_id"]].merge(
        tabular_source,
        on="image_id",
        how="left",
        indicator=True,
    )
    missing = merged[merged["_merge"] == "left_only"]
    if not missing.empty:
        examples = ", ".join(missing["image_id"].head(5).astype(str))
        raise ValueError(f"Missing tabular rows for image_id values: {examples}")
    return merged.drop(columns=["image_id", "_merge"])


def make_image_transform(image_size: int, training: bool):
    operations = [transforms.Resize((image_size, image_size))]
    if training:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(operations)


def build_model(
    model_name: str,
    tabular_dim: int,
    num_classes: int,
    args: argparse.Namespace,
):
    if model_name == "mlp":
        return MLPClassifier(
            input_dim=tabular_dim,
            num_classes=num_classes,
            hidden_dims=(128, 64),
            dropout=args.dropout,
        )
    if model_name == "resnet":
        return ImageClassifier(
            num_classes=num_classes,
            pretrained=args.pretrained_resnet,
            freeze_backbone=args.freeze_backbone,
            dropout=args.dropout,
        )
    if model_name == "early_fusion":
        return EarlyFusionClassifier(
            tabular_dim=tabular_dim,
            num_classes=num_classes,
            pretrained=args.pretrained_resnet,
            freeze_backbone=args.freeze_backbone,
            tabular_hidden_dim=64,
            fusion_hidden_dim=128,
            dropout=args.dropout,
        )
    if model_name == "late_fusion":
        return LateFusionClassifier(
            tabular_dim=tabular_dim,
            num_classes=num_classes,
            pretrained=args.pretrained_resnet,
            freeze_backbone=args.freeze_backbone,
            tabular_hidden_dims=(128, 64),
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def make_datasets(
    model_name: str,
    train_manifest: pd.DataFrame,
    eval_manifest: pd.DataFrame,
    tabular_source: pd.DataFrame,
    label_to_index: Dict[int, int],
    args: argparse.Namespace,
):
    use_images = model_name in {"resnet", "early_fusion", "late_fusion"}
    use_tabular = model_name in {"mlp", "early_fusion", "late_fusion"}
    train_tabular = None
    eval_tabular = None
    tabular_dim = 0

    if use_tabular:
        train_raw = merge_tabular(train_manifest, tabular_source)
        eval_raw = merge_tabular(eval_manifest, tabular_source)
        feature_columns = [column for column in train_raw.columns if column not in SOURCE_LABEL_COLUMNS]
        preprocessor = TabularPreprocessor.fit(train_raw, feature_columns)
        train_tabular = preprocessor.transform(train_raw)
        eval_tabular = preprocessor.transform(eval_raw)
        tabular_dim = int(train_tabular.shape[1])
        if tabular_dim <= 0:
            raise ValueError("No tabular features remained after preprocessing")

    train_transform = make_image_transform(args.image_size, training=True) if use_images else None
    eval_transform = make_image_transform(args.image_size, training=False) if use_images else None
    return (
        RetinopathyDataset(
            train_manifest,
            label_to_index,
            image_transform=train_transform,
            tabular_matrix=train_tabular,
            use_images=use_images,
            use_tabular=use_tabular,
        ),
        RetinopathyDataset(
            eval_manifest,
            label_to_index,
            image_transform=eval_transform,
            tabular_matrix=eval_tabular,
            use_images=use_images,
            use_tabular=use_tabular,
        ),
        tabular_dim,
    )


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_class_weights(train_manifest: pd.DataFrame, label_values: Sequence[int]):
    counts = train_manifest[LABEL_COLUMN].value_counts().to_dict()
    total = float(len(train_manifest))
    weights = []
    for label in label_values:
        count = float(counts.get(label, 0))
        weights.append(total / (len(label_values) * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def batch_forward(model, batch: Dict[str, object], device):
    image = batch.get("image")
    tabular = batch.get("tabular")
    if image is not None:
        image = image.to(device, non_blocking=True)
    if tabular is not None:
        tabular = tabular.to(device, non_blocking=True).float()
    return model(image=image, tabular=tabular)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_amp: bool,
    max_batches: int | None,
) -> tuple[float, int, int]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    total_loss = 0.0
    total_examples = 0
    batch_count = 0
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch_count += 1
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with amp_context:
            logits = batch_forward(model, batch, device)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples), total_examples, batch_count


def binary_auc(y_true: Sequence[int], y_score: Sequence[float], positive_label: int) -> float:
    positives = [1 if value == positive_label else 0 for value in y_true]
    positive_count = sum(positives)
    negative_count = len(positives) - positive_count
    if positive_count == 0 or negative_count == 0:
        return math.nan

    order = sorted(range(len(y_score)), key=lambda index: y_score[index])
    ranks = [0.0] * len(y_score)
    cursor = 0
    while cursor < len(order):
        next_cursor = cursor + 1
        while (
            next_cursor < len(order)
            and y_score[order[next_cursor]] == y_score[order[cursor]]
        ):
            next_cursor += 1
        average_rank = (cursor + 1 + next_cursor) / 2.0
        for rank_index in range(cursor, next_cursor):
            ranks[order[rank_index]] = average_rank
        cursor = next_cursor

    positive_rank_sum = sum(rank for rank, is_positive in zip(ranks, positives) if is_positive)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    positive_label: int,
) -> Dict[str, float]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == positive_label and pred == positive_label)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth != positive_label and pred != positive_label)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != positive_label and pred == positive_label)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == positive_label and pred != positive_label)
    total = max(1, len(y_true))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "f1": f1,
        "auc": binary_auc(y_true, y_score, positive_label),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate(
    model,
    loader,
    device,
    label_values: Sequence[int],
    positive_label: int,
    max_batches: int | None,
) -> tuple[Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []
    prediction_rows: List[Dict[str, object]] = []
    positive_index = label_values.index(positive_label)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            logits = batch_forward(model, batch, device)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            predicted_indices = probabilities.argmax(axis=1)
            labels = batch["label"].detach().cpu().numpy()
            original_truth = [label_values[int(index)] for index in labels]
            original_pred = [label_values[int(index)] for index in predicted_indices]
            positive_scores = probabilities[:, positive_index].tolist()

            y_true.extend(original_truth)
            y_pred.extend(original_pred)
            y_score.extend(positive_scores)

            for row_index, (truth, pred, score) in enumerate(
                zip(original_truth, original_pred, positive_scores)
            ):
                prediction_rows.append(
                    {
                        "image_id": batch["image_id"][row_index],
                        "patient_id": batch["patient_id"][row_index],
                        "true_label": truth,
                        "pred_label": pred,
                        "score_positive": score,
                    }
                )
    return compute_metrics(y_true, y_pred, y_score, positive_label), prediction_rows


def metric_value(metrics: Dict[str, float], name: str) -> float:
    value = metrics[name]
    return -1.0 if math.isnan(value) else value


def train_and_evaluate(
    model_name: str,
    train_manifest: pd.DataFrame,
    eval_manifest: pd.DataFrame,
    tabular_source: pd.DataFrame,
    label_values: Sequence[int],
    args: argparse.Namespace,
    device,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    label_to_index = {label: index for index, label in enumerate(label_values)}
    train_dataset, eval_dataset, tabular_dim = make_datasets(
        model_name,
        train_manifest,
        eval_manifest,
        tabular_source,
        label_to_index,
        args,
    )
    model = build_model(
        model_name,
        tabular_dim=tabular_dim,
        num_classes=len(label_values),
        args=args,
    ).to(device)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = make_loader(eval_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_weights = make_class_weights(train_manifest, label_values).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")

    best_metrics: Dict[str, float] | None = None
    best_state = None
    best_epoch = 0
    train_examples_used = 0
    train_batches_used = 0
    optimizer_train_seconds = 0.0
    selection_eval_seconds = 0.0
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_train_start = time.perf_counter()
        train_loss, train_examples_used, train_batches_used = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp,
            args.max_train_batches,
        )
        optimizer_train_seconds += time.perf_counter() - epoch_train_start
        selection_eval_start = time.perf_counter()
        current_metrics, _ = evaluate(
            model,
            eval_loader,
            device,
            label_values,
            args.positive_label,
            args.max_eval_batches,
        )
        selection_eval_seconds += time.perf_counter() - selection_eval_start
        current_metrics["train_loss"] = train_loss
        if best_metrics is None or metric_value(current_metrics, args.select_metric) > metric_value(
            best_metrics,
            args.select_metric,
        ):
            best_metrics = current_metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[{model_name}] epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.4f} acc={current_metrics['accuracy']:.4f} "
            f"f1={current_metrics['f1']:.4f} auc={current_metrics['auc']:.4f}"
        )

    train_seconds = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)
    eval_start = time.perf_counter()
    final_metrics, predictions = evaluate(
        model,
        eval_loader,
        device,
        label_values,
        args.positive_label,
        args.max_eval_batches,
    )
    eval_seconds = time.perf_counter() - eval_start
    final_metrics["train_loss"] = best_metrics["train_loss"] if best_metrics else math.nan
    result = {
        "model": model_name,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "train_time_seconds": train_seconds,
        "optimizer_train_time_seconds": optimizer_train_seconds,
        "selection_eval_time_seconds": selection_eval_seconds,
        "eval_seconds": eval_seconds,
        "eval_time_seconds": eval_seconds,
        "test_time_seconds": eval_seconds,
        "total_time_seconds": train_seconds + eval_seconds,
        "train_rows": int(len(train_manifest)),
        "eval_rows": int(len(eval_manifest)),
        "train_examples_used": int(train_examples_used),
        "train_batches_used": int(train_batches_used),
        "eval_examples_used": int(len(predictions)),
        "tabular_dim": int(tabular_dim),
        **final_metrics,
    }
    return result, predictions


def finite_float_values(rows: Sequence[Dict[str, object]], column: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if column not in row:
            continue
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            values.append(value)
    return values


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = ["accuracy", "f1", "auc", "sensitivity", "specificity", "precision"]
    time_columns = [
        "train_seconds",
        "train_time_seconds",
        "optimizer_train_time_seconds",
        "selection_eval_time_seconds",
        "eval_seconds",
        "eval_time_seconds",
        "test_time_seconds",
        "total_time_seconds",
    ]
    summary_rows: List[Dict[str, object]] = []
    group_keys = sorted({(str(row["model"]), str(row.get("eval_split", "eval"))) for row in rows})
    for model_name, eval_split in group_keys:
        model_rows = [
            row
            for row in rows
            if str(row["model"]) == model_name and str(row.get("eval_split", "eval")) == eval_split
        ]
        summary: Dict[str, object] = {
            "model": model_name,
            "eval_split": eval_split,
            "runs": len(model_rows),
        }
        for metric in metrics:
            values = finite_float_values(model_rows, metric)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else math.nan
            summary[f"{metric}_std"] = float(np.std(values, ddof=0)) if values else math.nan
        for column in time_columns:
            values = finite_float_values(model_rows, column)
            summary[f"{column}_mean"] = float(np.mean(values)) if values else math.nan
            summary[f"{column}_std"] = float(np.std(values, ddof=0)) if values else math.nan
        summary_rows.append(summary)
    return summary_rows


def run_fold_models(
    fold_dirs: Sequence[Path],
    models: Sequence[str],
    tabular_source: pd.DataFrame,
    label_values: Sequence[int],
    args: argparse.Namespace,
    device,
    results_dir: Path,
) -> List[Dict[str, object]]:
    metric_rows: List[Dict[str, object]] = []
    prediction_dir = results_dir / "predictions"
    for fold_dir in fold_dirs:
        fold_number = int(fold_dir.name.split("_")[-1])
        train_manifest = read_manifest(fold_dir / "train.csv")
        eval_manifest = read_manifest(fold_dir / "val.csv")
        for model_name in models:
            print(f"[INFO] Running {model_name} on fold {fold_number}")
            result, predictions = train_and_evaluate(
                model_name,
                train_manifest,
                eval_manifest,
                tabular_source,
                label_values,
                args,
                device,
            )
            result["fold"] = fold_number
            result["eval_split"] = "val"
            metric_rows.append(result)
            write_csv(
                prediction_dir / f"{model_name}_fold_{fold_number}_val_predictions.csv",
                predictions,
            )
    return metric_rows


def run_final_test(
    models: Sequence[str],
    tabular_source: pd.DataFrame,
    label_values: Sequence[int],
    args: argparse.Namespace,
    device,
    results_dir: Path,
) -> List[Dict[str, object]]:
    split_root = resolve_path(args.split_root)
    train_manifest = read_manifest(split_root / "train.csv")
    test_manifest = read_manifest(split_root / "test.csv")
    metric_rows: List[Dict[str, object]] = []
    prediction_dir = results_dir / "predictions"
    for model_name in models:
        print(f"[INFO] Running final outer test for {model_name}")
        result, predictions = train_and_evaluate(
            model_name,
            train_manifest,
            test_manifest,
            tabular_source,
            label_values,
            args,
            device,
        )
        result["fold"] = "outer_train"
        result["eval_split"] = "test"
        metric_rows.append(result)
        write_csv(prediction_dir / f"{model_name}_outer_test_predictions.csv", predictions)
    return metric_rows


def validate_data_inputs(
    fold_dirs: Sequence[Path],
    tabular_source: pd.DataFrame,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
) -> None:
    outer_overlap = set(outer_train["patient_id"].astype(str)) & set(
        outer_test["patient_id"].astype(str)
    )
    if outer_overlap:
        examples = ", ".join(sorted(outer_overlap)[:5])
        raise RuntimeError(f"Outer train/test patient overlap: {examples}")
    merge_tabular(outer_train, tabular_source)
    merge_tabular(outer_test, tabular_source)
    print(
        "[DATA] outer train/test: "
        f"{len(outer_train)}/{len(outer_test)} images, "
        f"{outer_train['patient_id'].nunique()}/{outer_test['patient_id'].nunique()} patients"
    )

    for fold_dir in fold_dirs:
        fold_number = int(fold_dir.name.split("_")[-1])
        train_manifest = read_manifest(fold_dir / "train.csv")
        val_manifest = read_manifest(fold_dir / "val.csv")
        patient_overlap = set(train_manifest["patient_id"].astype(str)) & set(
            val_manifest["patient_id"].astype(str)
        )
        image_overlap = set(train_manifest["image_id"].astype(str)) & set(
            val_manifest["image_id"].astype(str)
        )
        if patient_overlap:
            examples = ", ".join(sorted(patient_overlap)[:5])
            raise RuntimeError(f"Fold {fold_number} train/val patient overlap: {examples}")
        if image_overlap:
            examples = ", ".join(sorted(image_overlap)[:5])
            raise RuntimeError(f"Fold {fold_number} train/val image overlap: {examples}")
        merge_tabular(train_manifest, tabular_source)
        merge_tabular(val_manifest, tabular_source)
        print(
            f"[DATA] fold {fold_number}: "
            f"train={len(train_manifest)} val={len(val_manifest)} "
            f"patients={train_manifest['patient_id'].nunique()}/"
            f"{val_manifest['patient_id'].nunique()} "
            f"val_counts={val_manifest[LABEL_COLUMN].value_counts().sort_index().to_dict()}"
        )


def main() -> int:
    run_start = time.perf_counter()
    run_started_at = datetime.now().isoformat(timespec="seconds")
    args = parse_args()
    models = normalize_models(args.models)

    split_root = resolve_path(args.split_root).resolve()
    tabular_csv = resolve_path(args.tabular_csv).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = (
        resolve_path(args.results_dir).resolve()
        if args.results_dir
        else split_root / "deep_baselines" / f"run_{timestamp}"
    )
    console_log_path = None
    if not args.no_console_log:
        console_log_path = (
            resolve_path(args.console_log).resolve()
            if args.console_log
            else results_dir / "console.log"
        )
        enable_console_logging(console_log_path)

    fold_dirs = find_fold_dirs(split_root, args.folds)
    if args.max_folds is not None:
        fold_dirs = fold_dirs[: args.max_folds]
    tabular_source = load_tabular_source(tabular_csv, args.tabular_columns)

    outer_train = read_manifest(split_root / "train.csv")
    outer_test = read_manifest(split_root / "test.csv")
    label_values = sorted(
        set(outer_train[LABEL_COLUMN].astype(int)).union(set(outer_test[LABEL_COLUMN].astype(int)))
    )
    if args.positive_label not in label_values:
        raise ValueError(f"--positive-label {args.positive_label} is not present in labels {label_values}")

    if args.validate_data_only:
        validate_data_inputs(fold_dirs, tabular_source, outer_train, outer_test)
        print("[OK] Data inputs are ready for MLP, ResNet, Early Fusion, and Late Fusion.")
        return 0

    require_dependencies(models)
    set_seed(args.seed)

    results_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    config = {
        "models": models,
        "split_root": str(split_root),
        "tabular_csv": str(tabular_csv),
        "folds": [int(path.name.split("_")[-1]) for path in fold_dirs],
        "label_values": label_values,
        "positive_label": args.positive_label,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pretrained_resnet": args.pretrained_resnet,
        "freeze_backbone": args.freeze_backbone,
        "run_final_test": args.run_final_test,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "select_metric": args.select_metric,
        "started_at": run_started_at,
        "console_log": str(console_log_path) if console_log_path else None,
    }
    with (results_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    metric_rows = run_fold_models(
        fold_dirs,
        models,
        tabular_source,
        label_values,
        args,
        device,
        results_dir,
    )
    if args.run_final_test:
        metric_rows.extend(run_final_test(models, tabular_source, label_values, args, device, results_dir))

    metrics_path = results_dir / "metrics_by_fold.csv"
    summary_path = results_dir / "summary.csv"
    write_csv(metrics_path, metric_rows)
    summary_rows = summarize_metrics(metric_rows)
    write_csv(summary_path, summary_rows)
    run_finished_at = datetime.now().isoformat(timespec="seconds")
    run_time_seconds = time.perf_counter() - run_start
    run_summary_path = results_dir / "run_summary.json"
    with run_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "started_at": run_started_at,
                "finished_at": run_finished_at,
                "run_time_seconds": run_time_seconds,
                "metrics_csv": str(metrics_path),
                "summary_csv": str(summary_path),
                "models": models,
                "folds": [int(path.name.split("_")[-1]) for path in fold_dirs],
                "run_final_test": args.run_final_test,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(f"[OK] Wrote metrics: {metrics_path}")
    print(f"[OK] Wrote summary: {summary_path}")
    print(f"[OK] Wrote run summary: {run_summary_path}")
    if console_log_path:
        print(f"[OK] Saved console log: {console_log_path}")
    for row in summary_rows:
        print(
            "{model} [{eval_split}]: acc={accuracy_mean:.4f} f1={f1_mean:.4f} "
            "auc={auc_mean:.4f} sens={sensitivity_mean:.4f} spec={specificity_mean:.4f} "
            "train={train_time_seconds_mean:.2f}s test={test_time_seconds_mean:.2f}s "
            "total={total_time_seconds_mean:.2f}s".format(**row)
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            raise SystemExit(1)
        raise
