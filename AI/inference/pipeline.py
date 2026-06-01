from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
from PIL import Image


STAGE1_CLASS_NAMES: tuple[str, ...] = ("NonGarbage", "Garbage")
GARBAGE_CLASSES: tuple[str, ...] = (
    "Clothes",
    "Glass",
    "Plastic",
    "Shoes",
    "Cardboard",
    "Paper",
    "Metal",
    "Battery",
    "Biological",
    "Trash",
)

RESIZE_SIZE = 256
CROP_SIZE = 224
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    is_garbage: bool
    stage1_label: str
    stage1_confidence: float
    stage2_label: str | None = None
    stage2_confidence: float | None = None
    stage2_model_count: int = 0
    ensemble_used: bool = False
    stage1_latency_ms: float | None = None
    stage2_latency_ms: float | None = None
    total_latency_ms: float | None = None

    def to_dict(self) -> dict[str, str | float | int | bool | None]:
        return asdict(self)


class OnnxImageClassifier:
    def __init__(
        self,
        model_path: str | Path,
        class_names: Iterable[str],
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required. Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.class_names = tuple(class_names)
        
        # Enable all graph optimizations (fusion, constant folding, etc.)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        logits = self.session.run([self.output_name], {self.input_name: batch})[0]
        return softmax(np.asarray(logits, dtype=np.float32), axis=1)


class HierarchicalGarbageClassifier:
    """Stage 1 garbage detection followed by Stage 2 classification.

    Stage 2 uses single-model inference by default. If multiple Stage 2 ONNX
    paths are provided, their class probabilities are averaged as an ensemble.
    """

    def __init__(
        self,
        stage1_model_path: str | Path,
        stage2_model_paths: str | Path | Iterable[str | Path],
        garbage_threshold: float = 0.5,
        providers: list[str] | None = None,
        use_stage2_tta: bool = False,
    ) -> None:
        if not 0.0 <= garbage_threshold <= 1.0:
            raise ValueError("garbage_threshold must be between 0.0 and 1.0")

        self.garbage_threshold = garbage_threshold
        self.use_stage2_tta = use_stage2_tta
        self.stage1 = OnnxImageClassifier(stage1_model_path, STAGE1_CLASS_NAMES, providers)
        self.stage2_models = [
            OnnxImageClassifier(model_path, GARBAGE_CLASSES, providers)
            for model_path in normalize_model_paths(stage2_model_paths)
        ]
        if not self.stage2_models:
            raise ValueError("At least one Stage 2 ONNX model path is required.")

    @property
    def ensemble_used(self) -> bool:
        return len(self.stage2_models) > 1

    def predict(self, image: str | Path | Image.Image) -> PredictionResult:
        started_at = perf_counter()
        pil_image = load_image(image)

        stage1_started_at = perf_counter()
        stage1_probabilities = self.stage1.predict_proba(preprocess_image(pil_image))[0]
        stage1_latency_ms = elapsed_ms(stage1_started_at)

        garbage_probability = float(stage1_probabilities[1])
        non_garbage_probability = float(stage1_probabilities[0])
        if garbage_probability < self.garbage_threshold:
            return PredictionResult(
                label="쓰레기가 아닙니다",
                confidence=non_garbage_probability,
                is_garbage=False,
                stage1_label="NonGarbage",
                stage1_confidence=non_garbage_probability,
                stage2_model_count=len(self.stage2_models),
                ensemble_used=self.ensemble_used,
                stage1_latency_ms=stage1_latency_ms,
                stage2_latency_ms=0.0,
                total_latency_ms=elapsed_ms(started_at),
            )

        stage2_started_at = perf_counter()
        stage2_probabilities = self._predict_stage2_probabilities(pil_image)
        stage2_latency_ms = elapsed_ms(stage2_started_at)

        class_index = int(np.argmax(stage2_probabilities))
        label = GARBAGE_CLASSES[class_index]
        confidence = float(stage2_probabilities[class_index])
        return PredictionResult(
            label=label,
            confidence=confidence,
            is_garbage=True,
            stage1_label="Garbage",
            stage1_confidence=garbage_probability,
            stage2_label=label,
            stage2_confidence=confidence,
            stage2_model_count=len(self.stage2_models),
            ensemble_used=self.ensemble_used,
            stage1_latency_ms=stage1_latency_ms,
            stage2_latency_ms=stage2_latency_ms,
            total_latency_ms=elapsed_ms(started_at),
        )

    def _predict_stage2_probabilities(self, image: Image.Image) -> np.ndarray:
        batches = make_batches(image, use_tta=self.use_stage2_tta)
        model_probabilities: list[np.ndarray] = []
        for model in self.stage2_models:
            tta_probabilities = [model.predict_proba(batch)[0] for batch in batches]
            model_probabilities.append(np.mean(tta_probabilities, axis=0))
        return np.mean(model_probabilities, axis=0)


def normalize_model_paths(model_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(model_paths, (str, Path)):
        return [Path(model_paths)]
    return [Path(model_path) for model_path in model_paths]


def load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def make_batches(image: Image.Image, use_tta: bool) -> list[np.ndarray]:
    batches = [preprocess_image(image)]
    if use_tta:
        batches.append(preprocess_image(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    return batches


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = square_pad(image.convert("RGB"))
    image = image.resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.BILINEAR)
    image = center_crop(image, CROP_SIZE)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - NORMALIZE_MEAN) / NORMALIZE_STD
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype(np.float32)


def square_pad(image: Image.Image) -> Image.Image:
    width, height = image.size
    max_side = max(width, height)
    padded = Image.new("RGB", (max_side, max_side), color=(0, 0, 0))
    left = (max_side - width) // 2
    top = (max_side - height) // 2
    padded.paste(image, (left, top))
    return padded


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - crop_size) // 2)
    top = max(0, (height - crop_size) // 2)
    return image.crop((left, top, left + crop_size, top + crop_size))


def softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hierarchical garbage classification with ONNX models.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--stage1-model", type=Path, required=True)
    parser.add_argument(
        "--stage2-model",
        type=Path,
        action="append",
        required=True,
        help="Pass once for single-model inference or multiple times for Stage 2 ensemble.",
    )
    parser.add_argument("--garbage-threshold", type=float, default=0.5)
    parser.add_argument("--use-stage2-tta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = HierarchicalGarbageClassifier(
        stage1_model_path=args.stage1_model,
        stage2_model_paths=args.stage2_model,
        garbage_threshold=args.garbage_threshold,
        use_stage2_tta=args.use_stage2_tta,
    )
    result = classifier.predict(args.image)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
