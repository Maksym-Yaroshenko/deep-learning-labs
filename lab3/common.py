"""Спільні функції для ЛР3: класи, датасет, перетворення, модель і режими навчання."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

IMG_SIZE = 160
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EPOCHS = 8
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 64
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4
LR_HEAD = 3e-3
LR_LAYER4 = 3e-4

TOP_LEVEL_MODULES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")

# Модулі, які повинні залишатися в eval() під час навчання
FROZEN_EVAL_MODULES = {
    "scratch": (),
    "frozen": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
    "unfreeze": ("conv1", "bn1", "layer1", "layer2", "layer3"),
}

# Очікувана кількість параметрів із requires_grad=True для кожного режиму
EXPECTED_TOTAL_PARAMS = 11_179_590
EXPECTED_TRAINABLE_PARAMS = {
    "scratch": 11_179_590,
    "frozen": 3_078,
    "unfreeze": 8_396_806,
}


def select_train_device(requested: str = "auto") -> torch.device:
    """Обирає пристрій для навчання: MPS / CUDA / CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dataset_dirs(data_dir: Path) -> tuple[Path, Path]:
    """Знаходить теки seg_train/seg_train та seg_test/seg_test усередині data_dir."""
    train_dir = data_dir / "seg_train" / "seg_train"
    test_dir = data_dir / "seg_test" / "seg_test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Не знайдено очікувану структуру датасету в {data_dir.resolve()}.\n"
            "Очікуються теки:\n"
            "  <data_dir>/seg_train/seg_train/<клас>/*.jpg\n"
            "  <data_dir>/seg_test/seg_test/<клас>/*.jpg"
        )
    return train_dir, test_dir


def file_sha256(path: Path) -> str:
    """Обчислює SHA-256 хеш файлу."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split_rows(split_csv: Path, part: str | None = None) -> list[dict[str, str]]:
    """Читає рядки зі split.csv; якщо вказано part, фільтрує лише цю частину."""
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if part is not None:
        return [r for r in rows if r["part"] == part]
    return rows


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Повертає перетворення для train та для validation/test."""
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


class IntelSplitDataset(Dataset):
    """Завантажує зображення строго за списком рядків зі split.csv."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        data_dir: Path,
        transform: transforms.Compose,
    ) -> None:
        self.rows = rows
        self.train_dir, self.test_dir = resolve_dataset_dirs(data_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def image_path(self, index: int) -> Path:
        row = self.rows[index]
        base = self.train_dir if row["part"] == "train" else self.test_dir
        return base / row["label"] / row["file"]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.image_path(index)
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        tensor = self.transform(rgb)
        label = CLASS_TO_IDX[row["label"]]
        return tensor, label


def set_train_mode(model: torch.nn.Module, regime: str) -> None:
    """Переводить модель у train(), а заморожені модулі режиму — назад у eval()."""
    model.train()
    for module_name in FROZEN_EVAL_MODULES[regime]:
        getattr(model, module_name).eval()


def build_model_and_optimizer(
    regime: str, seed: int, device: torch.device
) -> tuple[torch.nn.Module, torch.optim.AdamW]:
    """Будує ResNet-18 і AdamW у строго фіксованому порядку кроків 1–5."""
    # Крок 1: фіксація зерен
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Крок 2: створення ResNet-18 із потрібними вагами
    if regime == "scratch":
        weights = None
    elif regime in ("frozen", "unfreeze"):
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Невідомий режим: {regime}")

    model = models.resnet18(weights=weights)

    # Крок 3: заміна голови на 6 класів зі стандартною ініціалізацією
    model.fc = torch.nn.Linear(512, len(CLASSES))

    # Крок 4: виставлення requires_grad згідно з режимом
    if regime == "scratch":
        for p in model.parameters():
            p.requires_grad = True
    elif regime == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    elif regime == "unfreeze":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

    # Крок 5: перенесення на пристрій (float32) і створення AdamW лише з активних параметрів
    model = model.to(device=device, dtype=torch.float32)

    if regime == "scratch":
        param_groups = [{"params": list(model.parameters()), "lr": LR_HEAD}]
    elif regime == "frozen":
        param_groups = [{"params": list(model.fc.parameters()), "lr": LR_HEAD}]
    elif regime == "unfreeze":
        param_groups = [
            {"params": list(model.fc.parameters()), "lr": LR_HEAD},
            {"params": list(model.layer4.parameters()), "lr": LR_LAYER4},
        ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def build_eval_model(
    state_dict: dict[str, torch.Tensor], device: torch.device
) -> torch.nn.Module:
    """Створює ResNet-18 (6 класів) і завантажує збережені ваги для інференсу."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Повертає сирі логіти (N, 6) та істинні мітки (N,) у режимі eval()."""
    model.eval()
    all_logits = []
    all_targets = []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(y.numpy())
    return (
        np.concatenate(all_logits, axis=0).astype(np.float32),
        np.concatenate(all_targets, axis=0).astype(np.int64),
    )