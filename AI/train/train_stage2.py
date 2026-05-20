from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI.train.utils import resolve_device, seed_everything

MODEL_CHOICES = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
)


def import_training_dependencies() -> None:
    global ConfusionMatrixDisplay
    global DataLoader
    global GARBAGE_CLASSES
    global GarbageStage2Dataset
    global Subset
    global accuracy_score
    global confusion_matrix
    global f1_score
    global models
    global nn
    global plt
    global random_split
    global torch
    global train_test_split
    global train_transform
    global tqdm
    global val_test_transform

    try:
        import torch
        matplotlib_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
        from torch import nn
        from torch.utils.data import DataLoader, Subset, random_split
        from torchvision import models
        from tqdm import tqdm

        from AI.preprocessing.transform import train_transform, val_test_transform
        from AI.train import GARBAGE_CLASSES, GarbageStage2Dataset
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them from the project root with:\n"
            "  pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight transfer-learning model for Stage 2 garbage classification."
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "AI/train/runs/stage2")
    parser.add_argument("--model-name", choices=MODEL_CHOICES, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debug limit. Uses a random subset before train/validation split.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the classifier head. By default, the full model is fine-tuned.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights. Transfer learning uses pretrained weights by default.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    return parser.parse_args()


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


def plot_confusion_matrix(labels: list[int], predictions: list[int], output_path: Path) -> None:
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(GARBAGE_CLASSES))))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(GARBAGE_CLASSES),
    )
    _, axis = plt.subplots(figsize=(10, 10))
    display.plot(ax=axis, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Stage 2 Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    import_training_dependencies()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"{args.model_name}-{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(run_dir),
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        freeze_backbone=args.freeze_backbone,
        pretrained=not args.no_pretrained,
        device=str(device),
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    base_dataset = GarbageStage2Dataset(args.dataset_dir)
    base_indices = list(range(len(base_dataset)))
    if args.max_samples is not None:
        if args.max_samples < 2:
            raise ValueError("--max-samples must be at least 2.")
        sample_count = min(args.max_samples, len(base_indices))
        base_indices = random.Random(args.seed).sample(base_indices, sample_count)

    labels = [base_dataset.samples[index][1] for index in base_indices]
    train_positions, val_positions = split_indices(labels, args.val_ratio, args.seed)
    train_indices = [base_indices[index] for index in train_positions]
    val_indices = [base_indices[index] for index in val_positions]

    train_dataset = Subset(GarbageStage2Dataset(args.dataset_dir, transform=train_transform), train_indices)
    val_dataset = Subset(GarbageStage2Dataset(args.dataset_dir, transform=val_test_transform), val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        model_name=args.model_name,
        num_classes=len(GARBAGE_CLASSES),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float]] = []
    best_val_accuracy = 0.0
    best_predictions: list[int] = []
    best_labels: list[int] = []

    print(f"Training {args.model_name} on {device} with {len(train_dataset)} train / {len(val_dataset)} val images.")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_loss, val_acc, val_f1, val_labels, val_predictions = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
        }
        history.append(row)
        save_history_csv(history, run_dir / "metrics.csv")
        plot_training_curves(history, run_dir / "training_curves.png")

        print(
            "train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            "val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} val_f1={val_f1:.4f}".format(**row)
        )

        if val_acc >= best_val_accuracy:
            best_val_accuracy = val_acc
            best_labels = val_labels
            best_predictions = val_predictions
            torch.save(
                {
                    "model_name": args.model_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": GARBAGE_CLASSES,
                    "config": asdict(config),
                    "metrics": row,
                },
                run_dir / "best_model.pt",
            )

    plot_confusion_matrix(best_labels, best_predictions, run_dir / "confusion_matrix.png")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": max(history, key=lambda row: row["val_accuracy"])["epoch"],
                "final_metrics": history[-1],
                "artifacts": {
                    "checkpoint": "best_model.pt",
                    "metrics": "metrics.csv",
                    "training_curves": "training_curves.png",
                    "confusion_matrix": "confusion_matrix.png",
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDone. Results saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
