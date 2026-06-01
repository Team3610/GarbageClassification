from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Stage 2 single-model and optional ensemble ONNX inference."
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        required=True,
        help="Stage 2 ONNX model path. Pass once for single model or multiple times for ensemble.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-tta", action="store_true", help="Average original and horizontal-flip predictions.")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from sklearn.metrics import accuracy_score, f1_score

        from AI.inference.pipeline import GARBAGE_CLASSES, OnnxImageClassifier, load_image, make_batches
        from AI.train.datasets import GarbageStage2Dataset
    except ImportError as exc:
        raise SystemExit(
            "Benchmark dependencies are missing. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    dataset = GarbageStage2Dataset(args.dataset_dir, return_path=True)
    samples = dataset.samples[: args.max_samples] if args.max_samples else dataset.samples
    models = [OnnxImageClassifier(model_path, GARBAGE_CLASSES) for model_path in args.model]

    labels: list[int] = []
    predictions: list[int] = []
    latencies_ms: list[float] = []
    rows: list[dict[str, str | int | float | bool]] = []

    for image_path, label in samples:
        image = load_image(image_path)
        batches = make_batches(image, use_tta=args.use_tta)

        started_at = perf_counter()
        model_probabilities = []
        for model in models:
            batch_probabilities = [model.predict_proba(batch)[0] for batch in batches]
            model_probabilities.append(np.mean(batch_probabilities, axis=0))
        probabilities = np.mean(model_probabilities, axis=0)
        latency_ms = (perf_counter() - started_at) * 1000.0

        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        labels.append(label)
        predictions.append(prediction)
        latencies_ms.append(latency_ms)
        rows.append(
            {
                "path": str(image_path),
                "label": GARBAGE_CLASSES[label],
                "prediction": GARBAGE_CLASSES[prediction],
                "correct": label == prediction,
                "confidence": confidence,
                "latency_ms": latency_ms,
            }
        )

    print(f"models={len(models)} ensemble_used={len(models) > 1} tta={args.use_tta} samples={len(samples)}")
    print(f"accuracy={accuracy_score(labels, predictions):.4f}")
    print(f"macro_f1={f1_score(labels, predictions, average='macro', zero_division=0):.4f}")
    print(f"mean_latency_ms={float(np.mean(latencies_ms)):.2f}")
    print(f"p95_latency_ms={float(np.percentile(latencies_ms, 95)):.2f}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["path", "label", "prediction", "correct", "confidence", "latency_ms"],
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
