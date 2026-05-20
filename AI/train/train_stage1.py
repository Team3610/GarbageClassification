from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI.train.utils import resolve_device, seed_everything


STAGE1_CLASS_NAMES = ("NonGarbage", "Garbage")


def import_training_dependencies() -> None:
    global DataLoader
    global GarbageStage1Dataset
    global Subset
    global TrainConfig
    global build_model
    global nn
    global plot_confusion_matrix
    global plot_training_curves
    global run_epoch
    global save_history_csv
    global split_indices
    global torch
    global train_transform
    global val_test_transform

    try:
        matplotlib_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset

        from AI.preprocessing.transform import train_transform, val_test_transform
        from AI.train import GarbageStage1Dataset
        from AI.train.trainer import (
            TrainConfig,
            build_model,
            plot_confusion_matrix,
            plot_training_curves,
            run_epoch,
            save_history_csv,
            split_indices,
        )
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them from the project root with:\n"
            "  pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight binary classifier for Stage 1 (garbage vs non-garbage)."
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "AI/train/runs/stage1")
    parser.add_argument(
        "--model-name",
        choices=(
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "efficientnet_b0",
            "efficientnet_b1",
        ),
        default="mobilenet_v3_small",
    )
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

    base_dataset = GarbageStage1Dataset(args.dataset_dir)
    label_counts = Counter(label for _, label in base_dataset.samples)
    if label_counts.get(0, 0) == 0:
        raise SystemExit(
            "Stage 1 학습에는 non-garbage 이미지가 필요합니다.\n"
            f"'{args.dataset_dir}' 아래 'non_garbage/', 'not_garbage/', 'non_waste/' 중 한 폴더에 "
            "비쓰레기 이미지를 추가하세요."
        )
    print(
        f"Loaded {len(base_dataset)} samples "
        f"(non_garbage={label_counts.get(0, 0)}, garbage={label_counts.get(1, 0)})."
    )

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

    train_dataset = Subset(GarbageStage1Dataset(args.dataset_dir, transform=train_transform), train_indices)
    val_dataset = Subset(GarbageStage1Dataset(args.dataset_dir, transform=val_test_transform), val_indices)

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
        num_classes=len(STAGE1_CLASS_NAMES),
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
                    "class_names": STAGE1_CLASS_NAMES,
                    "config": asdict(config),
                    "metrics": row,
                },
                run_dir / "best_model.pt",
            )

    plot_confusion_matrix(
        best_labels,
        best_predictions,
        class_names=STAGE1_CLASS_NAMES,
        output_path=run_dir / "confusion_matrix.png",
        title="Stage 1 Validation Confusion Matrix",
    )
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
