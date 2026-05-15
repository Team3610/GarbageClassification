import argparse
import shutil
import zipfile
from pathlib import Path

# 데이터셋 자동 다운로드 - Dataset/README.md 참고

DATASET_FILE_ID = "1L8TpC9F72u0hcoA-kqcD3gvvgZNQvrPn"
EXPECTED_CLASS_DIRS = (
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the garbage classification dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Dataset"),
        help="Directory where dataset class folders will be extracted.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("Dataset/dataset.zip"),
        help="Path for the downloaded or existing dataset zip file.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an existing zip file instead of downloading from Google Drive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing class folders before extracting.",
    )
    return parser.parse_args()


def download_from_google_drive(file_id: str, output_path: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required to download from Google Drive.\n"
            "Install it with: pip install gdown\n"
            "Or download dataset.zip manually and run with --skip-download."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(output_path), quiet=False)


def clear_existing_class_dirs(dataset_dir: Path) -> None:
    for class_dir in EXPECTED_CLASS_DIRS:
        path = dataset_dir / class_dir
        if path.exists():
            shutil.rmtree(path)


def extract_zip(zip_path: Path, dataset_dir: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset zip file not found: {zip_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_dir)


def find_missing_class_dirs(dataset_dir: Path) -> list[str]:
    return [
        class_dir
        for class_dir in EXPECTED_CLASS_DIRS
        if not (dataset_dir / class_dir).is_dir()
    ]


def main() -> None:
    args = parse_args()

    if args.force:
        clear_existing_class_dirs(args.dataset_dir)

    if not args.skip_download:
        download_from_google_drive(DATASET_FILE_ID, args.zip_path)

    extract_zip(args.zip_path, args.dataset_dir)

    missing_dirs = find_missing_class_dirs(args.dataset_dir)
    if missing_dirs:
        missing = ", ".join(missing_dirs)
        raise SystemExit(f"Dataset extracted, but missing class folders: {missing}")

    print(f"Dataset is ready at: {args.dataset_dir.resolve()}")


if __name__ == "__main__":
    main()
