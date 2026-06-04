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

CLASS_TO_IDX: dict[str, int] = {
    class_name: index for index, class_name in enumerate(GARBAGE_CLASSES)
}
IDX_TO_CLASS: dict[int, str] = {
    index: class_name for class_name, index in CLASS_TO_IDX.items()
}
FOLDER_TO_CLASS: dict[str, str] = {
    class_name.lower(): class_name for class_name in GARBAGE_CLASSES
}
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
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
            
        if self.mode == "sigmoid":
            import torch
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
        if self.mode == "stage1":
            return {"NonGarbage": 0, "Garbage": 1}
        return CLASS_TO_IDX.copy()

    def _load_samples(self, non_garbage_dirs: tuple[str, ...]) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        for folder_name, class_name in FOLDER_TO_CLASS.items():
            label = 1 if self.mode == "stage1" else CLASS_TO_IDX[class_name]
            samples.extend(self._collect_images(self.root_dir / folder_name, label))

        if self.mode in {"stage1", "sigmoid"}:
            non_garbage_label = 0 if self.mode == "stage1" else 10
            for folder_name in non_garbage_dirs:
                samples.extend(self._collect_images(self.root_dir / folder_name, non_garbage_label))

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
    """Binary dataset: non-garbage=0, garbage=1."""

    def __init__(self, root_dir: str | Path, **kwargs) -> None:
        super().__init__(root_dir=root_dir, mode="stage1", **kwargs)


class GarbageStage2Dataset(GarbageDataset):
    """10-class garbage classification dataset."""

    def __init__(self, root_dir: str | Path, **kwargs) -> None:
        super().__init__(root_dir=root_dir, mode="stage2", **kwargs)
