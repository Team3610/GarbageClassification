from pathlib import Path
from typing import Callable, Iterable, Literal

from PIL import Image
from torch.utils.data import Dataset


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
DEFAULT_NON_GARBAGE_DIRS: tuple[str, ...] = (
    "non_garbage",
    "not_garbage",
    "non_waste",
)


class GarbageDataset(Dataset):
    """PyTorch Dataset for Stage 1 and Stage 2 garbage classification.

    Stage 1:
        - non-garbage folders -> 0
        - 10 garbage class folders -> 1

    Stage 2:
        - 10 garbage class folders -> fixed class index
        
    sigmoid:
        - 10 garbage class folders -> One-hot vector of class index (length 10)
        - non-garbage folders -> Zero vector (length 10)

    Args:
        root_dir (str | Path): 클래스 폴더가 들어 있는 데이터셋 루트 경로.
        mode (Literal["stage1", "stage2", "sigmoid"]): 라벨을 생성할 학습 모드.
        transform (Callable | None): 이미지에 적용할 torchvision transform.
        target_transform (Callable | None): 정수 라벨에 적용할 후처리 함수.
        non_garbage_dirs (Iterable[str]): 비쓰레기 이미지가 들어 있는 폴더명 목록.
        return_path (bool): 샘플 반환 시 이미지 경로를 함께 포함할지 여부.

    Raises:
        ValueError: 지원하지 않는 mode가 들어온 경우.
        FileNotFoundError: 지정한 mode에서 사용할 이미지 샘플을 찾지 못한 경우.
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
        """Dataset에 포함된 이미지 샘플 수를 반환한다.

        Returns:
            int: self.samples에 저장된 샘플 개수.
        """

        return len(self.samples)

    def __getitem__(self, index: int):
        """index에 해당하는 이미지와 라벨을 PyTorch Dataset 형식으로 반환한다.

        Args:
            index (int): 가져올 샘플 index.

        Returns:
            tuple: return_path가 False이면 (image, label), True이면 (image, label, path).
        """

        image_path, label = self.samples[index]
        # 모든 입력을 RGB 3채널로 고정해 grayscale/alpha 채널 이미지가 섞여도 transform shape가 일정하게 유지된다.
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
            
        if self.mode == "sigmoid":
            import torch

            # non-garbage는 어떤 쓰레기 클래스에도 속하지 않으므로 all-zero target으로 표현한다.
            # CrossEntropy용 정수 라벨과 다르게 sigmoid 계열 모델은 클래스별 독립 확률을 학습한다.
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
        """현재 mode에서 사용하는 클래스명과 라벨 index 매핑을 반환한다.

        Returns:
            dict[str, int]: Stage 1이면 이진 라벨 매핑, 그 외에는 10-class 매핑.
        """

        if self.mode == "stage1":
            return {"NonGarbage": 0, "Garbage": 1}
        return CLASS_TO_IDX.copy()

    def _load_samples(self, non_garbage_dirs: tuple[str, ...]) -> list[tuple[Path, int]]:
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
        if not directory.is_dir():
            return []

        return [
            (path, label)
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]


class GarbageStage1Dataset(GarbageDataset):
    """Binary dataset: non-garbage=0, garbage=1.

    Args:
        root_dir (str | Path): 클래스 폴더가 들어 있는 데이터셋 루트 경로.
        **kwargs: GarbageDataset으로 전달할 transform, return_path 등 추가 옵션.
    """

    def __init__(self, root_dir: str | Path, **kwargs) -> None:
        super().__init__(root_dir=root_dir, mode="stage1", **kwargs)


class GarbageStage2Dataset(GarbageDataset):
    """10-class garbage classification dataset.

    Args:
        root_dir (str | Path): 클래스 폴더가 들어 있는 데이터셋 루트 경로.
        **kwargs: GarbageDataset으로 전달할 transform, return_path 등 추가 옵션.
    """

    def __init__(self, root_dir: str | Path, **kwargs) -> None:
        super().__init__(root_dir=root_dir, mode="stage2", **kwargs)
