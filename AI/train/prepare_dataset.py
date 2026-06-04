import argparse
import shutil
import zipfile
from pathlib import Path

# Google Drive 공유 파일 ID는 바뀔 수 있으므로 Dataset/README의 데이터셋 출처와 함께 관리한다.

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
    """데이터셋 다운로드/압축 해제 CLI 옵션을 파싱한다.

    Returns:
        argparse.Namespace: dataset_dir, zip_path, skip_download, force 옵션.
    """

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
    """수동 다운로드와 자동 다운로드를 같은 zip 경로 규칙으로 맞춰 후속 압축 해제를 단순화한다.

    Args:
        file_id (str): Google Drive 공유 파일 ID.
        output_path (Path): 다운로드한 zip 파일을 저장할 경로.

    Returns:
        None

    Raises:
        SystemExit: gdown이 설치되어 있지 않은 경우.
    """

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
    """--force 실행 시 zip 안의 클래스 폴더만 지워 로컬 메모나 zip 파일은 보존한다.

    Args:
        dataset_dir (Path): Dataset 루트 경로.

    Returns:
        None
    """

    for class_dir in EXPECTED_CLASS_DIRS:
        path = dataset_dir / class_dir
        if path.exists():
            # Dataset/ 전체를 지우지 않는 이유는 dataset.zip이나 README 같은 보조 파일을 유지하기 위해서다.
            shutil.rmtree(path)


def extract_zip(zip_path: Path, dataset_dir: Path) -> None:
    """데이터셋 zip 파일을 Dataset 루트에 압축 해제한다.

    Args:
        zip_path (Path): 압축 해제할 zip 파일 경로.
        dataset_dir (Path): 압축을 풀 대상 Dataset 루트 경로.

    Returns:
        None

    Raises:
        FileNotFoundError: zip_path에 파일이 없는 경우.
    """

    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset zip file not found: {zip_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    # zip 내부 폴더 구조가 Dataset/<class_name>/... 형태라는 전제에서 바로 풀어낸다.
    # 구조가 바뀌면 아래 find_missing_class_dirs 검증에서 실패하게 둔다.
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_dir)


def find_missing_class_dirs(dataset_dir: Path) -> list[str]:
    """학습 코드가 기대하는 10개 폴더가 모두 준비됐는지 빠르게 검증한다.

    Args:
        dataset_dir (Path): Dataset 루트 경로.

    Returns:
        list[str]: 존재하지 않는 필수 클래스 폴더명 목록.
    """

    return [
        class_dir
        for class_dir in EXPECTED_CLASS_DIRS
        if not (dataset_dir / class_dir).is_dir()
    ]


def main() -> None:
    """CLI 엔트리포인트로 데이터셋 다운로드, 압축 해제, 폴더 검증을 수행한다.

    Returns:
        None
    """

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
