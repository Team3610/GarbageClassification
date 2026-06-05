from pathlib import Path
from typing import Callable, Iterable, Literal

from PIL import Image
from torch.utils.data import Dataset


# // 한국 환경부 분리배출 가이드라인 및 TACO 데이터셋 분류 기준을 참고하여 선정한 실생활 10대 쓰레기 핵심 분류 카테고리입니다.
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

# 이 tuple의 순서가 Stage 2 학습 label index와 추론 label index의 기준이다.
# 새 클래스를 추가하거나 순서를 바꾸면 기존 checkpoint/ONNX 모델의 출력 해석도 함께 바뀐다.
CLASS_TO_IDX: dict[str, int] = {
    class_name: index for index, class_name in enumerate(GARBAGE_CLASSES)
}
IDX_TO_CLASS: dict[int, str] = {
    index: class_name for class_name, index in CLASS_TO_IDX.items()
}
FOLDER_TO_CLASS: dict[str, str] = {
    class_name.lower(): class_name for class_name in GARBAGE_CLASSES
}
# 폴더명은 소문자, 모델 출력 라벨은 대문자 시작 표기를 사용하므로 한 곳에서만 변환한다.
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# // 학습 기여도를 높이기 위해 사용자가 폴더명을 non_garbage, not_garbage, non_waste 등으로 지정하여 수집한 비쓰레기 데이터를 모두 식별할 수 있도록 다중 폴더명을 허용합니다.
DEFAULT_NON_GARBAGE_DIRS: tuple[str, ...] = (
    "non_garbage",
    "not_garbage",
    "non_waste",
)


class GarbageDataset(Dataset):
    """
    /**
     * Stage 1(이진 분류), Stage 2(10클래스 분류) 및 Sigmoid(10클래스 멀티레이블 분류)용 PyTorch Dataset 클래스입니다.
     */
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: Literal["stage1", "stage2", "sigmoid"] = "stage2",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        non_garbage_dirs: Iterable[str] = DEFAULT_NON_GARBAGE_DIRS,
        return_path: bool = False,
    ) -> None:
        """
        /**
         * GarbageDataset 클래스를 초기화합니다.
         * @param {str | Path} root_dir - 데이터셋의 루트 경로
         * @param {Literal["stage1", "stage2", "sigmoid"]} mode - 학습 파이프라인 단계 (stage1, stage2, sigmoid)
         * @param {Callable | None} transform - 이미지 텐서 변환 함수
         * @param {Callable | None} target_transform - 타겟 라벨 변환 함수
         * @param {Iterable[str]} non_garbage_dirs - 비쓰레기 데이터 디렉토리 목록
         * @param {bool} return_path - 파일 경로 반환 여부
         * @returns {None}
         */
        """
        if mode not in {"stage1", "stage2", "sigmoid"}:
            raise ValueError("mode must be 'stage1', 'stage2', or 'sigmoid'")

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path
        self.samples = self._load_samples(tuple(non_garbage_dirs))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found for mode='{mode}' under '{self.root_dir}'. "
                f"Expected folders: {', '.join(FOLDER_TO_CLASS.keys())}"
            )

    def __len__(self) -> int:
        """
        /**
         * 데이터셋의 전체 샘플 개수를 반환합니다.
         * @returns {int} 샘플 개수
         */
        """
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        /**
         * 인덱스에 해당하는 샘플(이미지 및 라벨)을 반환합니다.
         * @param {int} index - 샘플 인덱스
         * @returns {tuple[torch.Tensor, int | torch.Tensor] | tuple[torch.Tensor, int | torch.Tensor, str]} 이미지, 라벨 (필요 시 파일경로 추가)
         */
        """
        image_path, label = self.samples[index]
        # 모든 입력을 RGB 3채널로 고정해 grayscale/alpha 채널 이미지가 섞여도 transform shape가 일정하게 유지된다.
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
            
        if self.mode == "sigmoid":
            import torch
            # // sigmoid 모드(Multi-label)에서는 쓰레기가 아닌 이미지(label=10)는 모든 클래스가 0인 zero-vector로, 쓰레기인 이미지는 해당 클래스 인덱스만 1.0인 10차원 One-hot 벡터로 변환합니다.
            target = torch.zeros(10, dtype=torch.float32)
            if label < 10:
                target[label] = 1.0
            label = target
        elif self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_path:
            return image, label, str(image_path)
        return image, label

    @property
    def class_to_idx(self) -> dict[str, int]:
        """
        /**
         * 현재 모드에 매핑되는 클래스-인덱스 딕셔너리를 반환합니다.
         * @returns {dict[str, int]} 클래스명과 인덱스 쌍
         */
        """
        if self.mode == "stage1":
            return {"NonGarbage": 0, "Garbage": 1}
        return CLASS_TO_IDX.copy()

    def _load_samples(self, non_garbage_dirs: tuple[str, ...]) -> list[tuple[Path, int]]:
        """
        /**
         * 전체 클래스 디렉토리에서 이미지 경로와 라벨 목록을 스캔하여 로드합니다.
         * @param {tuple[str, ...]} non_garbage_dirs - 비쓰레기 데이터 디렉토리 튜플
         * @returns {list[tuple[Path, int]]} 이미지 경로와 라벨 튜플 리스트
         */
        """
        samples: list[tuple[Path, int]] = []

        for folder_name, class_name in FOLDER_TO_CLASS.items():
            # Stage 1은 10개 쓰레기 폴더를 하나의 양성 라벨로 묶고, Stage 2는 클래스 index를 유지한다.
            # 같은 폴더 구조를 두 학습 문제에 재사용하기 위한 분기라서 mode별 라벨 의미가 다르다.
            label = 1 if self.mode == "stage1" else CLASS_TO_IDX[class_name]
            samples.extend(self._collect_images(self.root_dir / folder_name, label))

        if self.mode in {"stage1", "sigmoid"}:
            # sigmoid 모드의 10은 __getitem__에서 all-zero vector로 바꾸기 위한 내부 sentinel이다.
            # 실제 모델 출력 클래스는 0~9뿐이므로 이 값이 그대로 loss에 들어가면 안 된다.
            non_garbage_label = 0 if self.mode == "stage1" else 10
            for folder_name in non_garbage_dirs:
                samples.extend(self._collect_images(self.root_dir / folder_name, non_garbage_label))

        # 파일 시스템 순서 차이로 train/validation split이 흔들리지 않도록 정렬된 목록을 반환한다.
        return sorted(samples, key=lambda sample: str(sample[0]))

    def _collect_images(self, directory: Path, label: int) -> list[tuple[Path, int]]:
        """
        /**
         * 특정 디렉토리 내 지원하는 확장자를 가진 이미지들을 재귀적으로 수집합니다.
         * @param {Path} directory - 스캔 대상 디렉토리
         * @param {int} label - 할당할 정수 라벨
         * @returns {list[tuple[Path, int]]} 매칭된 이미지 경로와 라벨 리스트
         */
        """
        if not directory.is_dir():
            return []

        return [
            (path, label)
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]


class GarbageStage1Dataset(GarbageDataset):
    """
    /**
     * Stage 1 이진 분류 데이터셋을 표현합니다.
     */
    """

    def __init__(self, root_dir: str | Path, **kwargs) -> None:
        super().__init__(root_dir=root_dir, mode="stage1", **kwargs)


class GarbageStage2Dataset(GarbageDataset):
    """
    /**
     * Stage 2 10개 클래스 쓰레기 분류 데이터셋을 표현합니다.
     */
    """

    def __init__(self, root_dir: str | Path, **kwargs) -> None:
        super().__init__(root_dir=root_dir, mode="stage2", **kwargs)
