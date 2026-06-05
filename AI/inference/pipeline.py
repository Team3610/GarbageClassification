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
# 이 순서는 Stage 2 모델의 출력 index와 직접 연결된다.
# 학습할 때 저장된 class_names와 순서가 달라지면 확률은 맞아도 라벨 해석이 틀어진다.
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
# PyTorch 학습 transform과 동일한 ImageNet 정규화 값을 써야 ONNX logits 분포가 맞는다.
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PredictionResult:
    """계층형 추론의 최종 결과와 stage별 판단 근거를 함께 담는 데이터 객체.

    Attributes:
        label (str): 사용자에게 보여줄 최종 라벨.
        confidence (float): 최종 라벨의 확률값.
        is_garbage (bool): Stage 1 기준 쓰레기 여부.
        stage1_label (str): Stage 1에서 선택된 라벨.
        stage1_confidence (float): Stage 1 선택 라벨의 확률값.
        stage2_label (str | None): Stage 2에서 선택된 쓰레기 클래스 라벨.
        stage2_confidence (float | None): Stage 2 선택 라벨의 확률값.
        stage2_model_count (int): Stage 2 추론에 사용한 ONNX 모델 수.
        ensemble_used (bool): Stage 2 앙상블 사용 여부.
        stage1_latency_ms (float | None): Stage 1 추론 시간(ms).
        stage2_latency_ms (float | None): Stage 2 추론 시간(ms).
        total_latency_ms (float | None): 전체 추론 시간(ms).
    """

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
        """API 응답이나 JSON 출력에 바로 사용할 수 있는 dict로 변환한다.

        Returns:
            dict[str, str | float | int | bool | None]: dataclass 필드를 보존한 결과 dict.
        """

        return asdict(self)


class OnnxImageClassifier:
    """ONNX Runtime 세션 생성과 logits 후처리를 캡슐화한 단일 이미지 분류기.

    Args:
        model_path (str | Path): 로드할 ONNX 모델 경로.
        class_names (Iterable[str]): 모델 출력 index와 같은 순서의 클래스 이름 목록.
        providers (list[str] | None): ONNX Runtime execution provider 목록.

    Raises:
        ImportError: onnxruntime이 설치되어 있지 않은 경우.
        FileNotFoundError: model_path에 ONNX 파일이 없는 경우.
    """

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
        # class_names는 현재 후처리 검증용으로 보관한다.
        # ONNX 모델 자체에는 라벨명이 없기 때문에 호출부가 넘긴 순서를 신뢰한다.
        
        # CLI 추론은 CPU 실행이 기본이므로 가능한 graph optimization을 모두 켜서 지연 시간을 줄인다.
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
        """NCHW float32 batch를 입력받아 클래스별 softmax 확률을 반환한다.

        Args:
            batch (np.ndarray): shape이 (N, 3, 224, 224)인 float32 입력 batch.

        Returns:
            np.ndarray: shape이 (N, class_count)인 클래스별 확률 배열.
        """

        logits = self.session.run([self.output_name], {self.input_name: batch})[0]
        # export된 모델 출력은 확률이 아니라 logits이므로 여기에서만 softmax를 적용한다.
        return softmax(np.asarray(logits, dtype=np.float32), axis=1)


class HierarchicalGarbageClassifier:
    """Stage 1 garbage detection followed by Stage 2 classification.

    Stage 2 uses single-model inference by default. If multiple Stage 2 ONNX
    paths are provided, their class probabilities are averaged as an ensemble.

    Args:
        stage1_model_path (str | Path): 비쓰레기/쓰레기 이진 분류 ONNX 모델 경로.
        stage2_model_paths (str | Path | Iterable[str | Path]): 10-class 쓰레기 분류 ONNX 모델 경로.
        garbage_threshold (float): Stage 1에서 Garbage로 판정할 최소 확률.
        providers (list[str] | None): ONNX Runtime execution provider 목록.
        use_stage2_tta (bool): Stage 2에서 원본/좌우 반전 TTA를 사용할지 여부.

    Raises:
        ValueError: threshold가 0~1 범위를 벗어나거나 Stage 2 모델 경로가 비어 있는 경우.
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
        """Stage 2에서 둘 이상의 모델을 평균내는지 반환한다.

        Returns:
            bool: Stage 2 모델 수가 2개 이상이면 True.
        """

        return len(self.stage2_models) > 1

    def predict(self, image: str | Path | Image.Image) -> PredictionResult:
        """이미지 한 장을 계층형 파이프라인으로 분류한다.

        Args:
            image (str | Path | Image.Image): 이미지 파일 경로 또는 PIL 이미지 객체.

        Returns:
            PredictionResult: 최종 라벨, stage별 confidence, latency를 포함한 추론 결과.
        """

        started_at = perf_counter()
        pil_image = load_image(image)

        stage1_started_at = perf_counter()
        stage1_probabilities = self.stage1.predict_proba(preprocess_image(pil_image))[0]
        stage1_latency_ms = elapsed_ms(stage1_started_at)

        garbage_probability = float(stage1_probabilities[1])
        non_garbage_probability = float(stage1_probabilities[0])
        if garbage_probability < self.garbage_threshold:
            # Stage 1에서 비쓰레기로 판단되면 Stage 2를 생략해 불필요한 10-class 오분류를 막는다.
            # 이 분기가 없으면 비쓰레기 이미지도 항상 Plastic, Paper 같은 쓰레기 라벨 중 하나로 강제 분류된다.
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
            # TTA 평균을 먼저 만든 뒤 모델 간 확률을 평균내야 각 모델의 가중치가 동일해진다.
            # 반대로 모든 batch 결과를 한꺼번에 평균내면 TTA 개수가 늘어난 모델이 더 큰 비중을 갖게 될 수 있다.
            tta_probabilities = [model.predict_proba(batch)[0] for batch in batches]
            model_probabilities.append(np.mean(tta_probabilities, axis=0))
        return np.mean(model_probabilities, axis=0)


def normalize_model_paths(model_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """단일 경로와 여러 경로 입력을 내부 처리용 Path 리스트로 통일한다.

    Args:
        model_paths (str | Path | Iterable[str | Path]): 단일 모델 경로 또는 모델 경로 iterable.

    Returns:
        list[Path]: Path 객체로 변환된 모델 경로 목록.
    """

    if isinstance(model_paths, (str, Path)):
        return [Path(model_paths)]
    return [Path(model_path) for model_path in model_paths]


def load_image(image: str | Path | Image.Image) -> Image.Image:
    """파일 경로 또는 PIL 이미지를 RGB PIL 이미지로 정규화한다.

    Args:
        image (str | Path | Image.Image): 이미지 파일 경로 또는 PIL 이미지 객체.

    Returns:
        Image.Image: RGB 모드의 PIL 이미지.
    """

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def make_batches(image: Image.Image, use_tta: bool) -> list[np.ndarray]:
    """Stage 2 TTA가 켜진 경우 원본과 좌우 반전 이미지를 같은 전처리 규칙으로 묶는다.

    Args:
        image (Image.Image): 전처리할 PIL 이미지.
        use_tta (bool): 좌우 반전 test-time augmentation 사용 여부.

    Returns:
        list[np.ndarray]: ONNX 모델 입력으로 사용할 NCHW batch 목록.
    """

    batches = [preprocess_image(image)]
    if use_tta:
        batches.append(preprocess_image(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    return batches


def preprocess_image(image: Image.Image) -> np.ndarray:
    """학습 검증 파이프라인과 같은 256 resize, 224 center crop, 정규화 순서를 재현한다.

    Args:
        image (Image.Image): 원본 PIL 이미지.

    Returns:
        np.ndarray: shape이 (1, 3, 224, 224)인 float32 NCHW batch.
    """

    image = square_pad(image.convert("RGB"))
    # 먼저 256으로 키운 뒤 224 중앙 crop을 하는 흐름은 torchvision 검증 transform과 맞춘 것이다.
    # 단순 224 resize보다 학습 때 본 입력 분포와 가까워 ONNX 추론 결과가 안정적이다.
    image = image.resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.BILINEAR)
    image = center_crop(image, CROP_SIZE)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - NORMALIZE_MEAN) / NORMALIZE_STD
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype(np.float32)


def square_pad(image: Image.Image) -> Image.Image:
    """세로/가로로 긴 사진도 center crop에서 대상이 덜 잘리도록 정사각형으로 패딩한다.

    Args:
        image (Image.Image): RGB PIL 이미지.

    Returns:
        Image.Image: 긴 변을 기준으로 검은 배경을 채운 정사각형 이미지.
    """

    width, height = image.size
    max_side = max(width, height)
    padded = Image.new("RGB", (max_side, max_side), color=(0, 0, 0))
    left = (max_side - width) // 2
    top = (max_side - height) // 2
    padded.paste(image, (left, top))
    return padded


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    """이미지 중앙을 기준으로 고정 크기 영역을 잘라낸다.

    Args:
        image (Image.Image): crop 대상 PIL 이미지.
        crop_size (int): 잘라낼 정사각형 한 변의 길이.

    Returns:
        Image.Image: 중앙 crop이 적용된 PIL 이미지.
    """

    width, height = image.size
    left = max(0, (width - crop_size) // 2)
    top = max(0, (height - crop_size) // 2)
    return image.crop((left, top, left + crop_size, top + crop_size))


def softmax(values: np.ndarray, axis: int) -> np.ndarray:
    """수치 안정성을 위해 max-shift를 적용한 softmax를 계산한다.

    Args:
        values (np.ndarray): logits 배열.
        axis (int): softmax를 적용할 축.

    Returns:
        np.ndarray: 입력과 같은 shape의 확률 배열.
    """

    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def elapsed_ms(started_at: float) -> float:
    """perf_counter 시작 시각으로부터 지난 시간을 ms 단위로 계산한다.

    Args:
        started_at (float): perf_counter()로 기록한 시작 시각.

    Returns:
        float: 경과 시간(ms).
    """

    return (perf_counter() - started_at) * 1000.0


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다.

    Returns:
        argparse.Namespace: 이미지 경로, 모델 경로, threshold, TTA 옵션.
    """

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
    """CLI 엔트리포인트로 계층형 추론을 실행하고 JSON 결과를 출력한다.

    Returns:
        None
    """

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
