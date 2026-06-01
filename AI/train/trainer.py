from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm


MODEL_CHOICES = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
    "squeezenet1_1",
    "efficientnet_lite0",
)


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: str
    output_dir: str
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    val_ratio: float
    seed: int
    num_workers: int
    max_samples: int | None
    freeze_backbone: bool
    pretrained: bool
    device: str


def split_indices(labels: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")

    dataset_size = len(labels)
    val_size = max(1, int(dataset_size * val_ratio))
    train_size = dataset_size - val_size
    if train_size < 1:
        raise ValueError("Dataset is too small to create train/validation splits.")

    indices = list(range(dataset_size))
    try:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_size,
            random_state=seed,
            stratify=labels,
        )
        return list(train_indices), list(val_indices)
    except ValueError:
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(
            indices,
            [train_size, val_size],
            generator=generator,
        )
        return list(train_subset.indices), list(val_subset.indices)


def build_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "squeezenet1_1":
        weights = models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        model = models.squeezenet1_1(weights=weights)
        in_features = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(in_features, num_classes, kernel_size=1)
        model.num_classes = num_classes
    elif model_name == "efficientnet_lite0":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for efficientnet_lite0. Install it with `pip install timm`."
            ) from exc
        model = timm.create_model("efficientnet_lite0", pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier")

    return model


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(dataloader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_labels, all_predictions


def save_history_csv(history: list[dict[str, float]], path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_f1",
        "val_loss",
        "val_accuracy",
        "val_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="train")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    labels: list[int],
    predictions: list[int],
    class_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(class_names),
    )
    figure_size = max(6, len(class_names))
    _, axis = plt.subplots(figsize=(figure_size, figure_size))
    display.plot(ax=axis, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
