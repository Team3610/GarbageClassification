import argparse
import shutil
import zipfile
from pathlib import Path

# 데이터셋 자동 다운로드 - Dataset/README.md 참고

# // 팀 내에서 공유/배포 및 공통 관리를 용이하게 하기 위해 구글 드라이브에 미리 패키징하여 업로드해 둔 데이터셋 압축파일(dataset.zip)의 공유용 고유 파일 식별자 ID입니다.
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
    """
    /**
     * CLI 인자를 파싱합니다.
     * @returns {argparse.Namespace} 파싱된 명령행 인수 객체
     */
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
    """
    /**
     * 구글 드라이브로부터 대용량 파일 식별자 ID를 이용해 데이터셋 압축파일을 다운로드합니다.
     * @param {str} file_id - 구글 드라이브 파일 ID
     * @param {Path} output_path - 저장할 로컬 파일 경로
     * @returns {None}
     */
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
    """
    /**
     * 추출 전 충돌 방지를 위해 기존에 존재하던 클래스 디렉토리들을 완전히 비웁니다.
     * @param {Path} dataset_dir - 데이터셋 디렉토리 경로
     * @returns {None}
     */
    """
    for class_dir in EXPECTED_CLASS_DIRS:
        path = dataset_dir / class_dir
        if path.exists():
            shutil.rmtree(path)


def extract_zip(zip_path: Path, dataset_dir: Path) -> None:
    """
    /**
     * 다운로드 혹은 준비된 zip 압축 파일을 대상 디렉토리에 풉니다.
     * @param {Path} zip_path - 압축 파일 경로
     * @param {Path} dataset_dir - 압축을 해제할 디렉토리 경로
     * @returns {None}
     */
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset zip file not found: {zip_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_dir)


def find_missing_class_dirs(dataset_dir: Path) -> list[str]:
    """
    /**
     * 압축 해제 후 요구되는 필수 클래스 디렉토리 중 누락된 곳이 있는지 탐색합니다.
     * @param {Path} dataset_dir - 데이터셋 디렉토리 경로
     * @returns {list[str]} 누락된 클래스 디렉토리명 리스트
     */
    """
    return [
        class_dir
        for class_dir in EXPECTED_CLASS_DIRS
        if not (dataset_dir / class_dir).is_dir()
    ]


def main() -> None:
    """
    /**
     * 데이터셋 준비 전 과정을 제어하는 진입 함수입니다.
     * @returns {None}
     */
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
