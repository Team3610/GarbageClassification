import argparse
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# // TACO(Trash Annotations in Context) 오픈소스 쓰레기 데이터셋 어노테이션 정보가 수록된 공식 원격 Github JSON 주소입니다.
TACO_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/taco/master/data/annotations.json"

# // 기본 쓰레기 수집 데이터셋 내에 Metal(금속) 샘플의 개수가 부족하여, 모델의 일반화(Generalization) 능력을 강화하고자 TACO 데이터셋의 세부 금속 항목들을 매핑용 카테고리로 지정했습니다.
METAL_CATEGORIES = {
    "Aluminium foil",
    "Aluminium can",
    "Aerosol",
    "Metal bottle cap",
    "Metal lid",
    "Pop tab",
    "Other metal",
    "Food can",
    "Drink can"
}


def parse_args():
    """
    /**
     * CLI 인자를 파싱합니다.
     * @returns {argparse.Namespace} 파싱된 명령행 인수 객체
     */
    """
    parser = argparse.ArgumentParser(description="Download Metal category images from the TACO dataset.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Dataset" / "metal")
    # // 이미지 수 제한의 기본값 150은 Flickr API 제한에 걸리지 않도록 하며, 타 클래스와의 데이터 균형(Data Imbalance)을 고려한 크기입니다.
    parser.add_argument("--max-samples", type=int, default=150, help="Maximum number of images to download")
    return parser.parse_args()


def get_direct_flickr_image_url(page_url):
    """
    /**
     * Flickr 사진 웹페이지 HTML에서 실제 정적 이미지 경로(JPG/PNG)를 파싱 및 해결하여 반환합니다.
     * @param {str} page_url - Flickr 사진 페이지 URL
     * @returns {str | None} 해결된 정적 이미지의 직통 URL (실패 시 None)
     */
    """
    if "staticflickr.com" in page_url or page_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return page_url

    try:
        req = urllib.request.Request(
            page_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        match = re.search(r'<meta\s+property="og:image"\s+content="([^"]+)"', html)
        if match:
            return match.group(1)
        
        match_static = re.search(r'https://[a-z0-9\.]+\.staticflickr\.com/[^"\']+', html)
        if match_static:
            return match_static.group(0)
            
    except Exception as e:
        print(f"Error resolving Flickr page {page_url}: {e}")
    return None


def download_image(url, output_path):
    """
    /**
     * 직통 이미지 URL로부터 리소스를 스트림하여 지정한 경로에 저장합니다.
     * @param {str} url - 이미지 직통 URL
     * @param {Path} output_path - 저장할 로컬 파일 경로
     * @returns {bool} 성공 여부
     */
    """
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            output_path.write_bytes(response.read())
        return True
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
        return False


def main():
    """
    /**
     * TACO 메타데이터 다운로드부터 금속류 사진 파싱 및 로컬 다운로드 과정을 실행하는 메인 루프입니다.
     * @returns {None}
     */
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading TACO annotations.json...")
    try:
        with urllib.request.urlopen(TACO_ANNOTATIONS_URL) as response:
            annotations_data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Failed to download annotations file: {e}")
        return

    categories = annotations_data.get("categories", [])
    metal_category_ids = set()
    for cat in categories:
        if cat["name"] in METAL_CATEGORIES:
            metal_category_ids.add(cat["id"])
            print(f"Mapped TACO category: {cat['name']} (ID: {cat['id']})")

    annotations = annotations_data.get("annotations", [])
    metal_image_ids = set()
    for ann in annotations:
        if ann["category_id"] in metal_category_ids:
            metal_image_ids.add(ann["image_id"])

    print(f"Found {len(metal_image_ids)} images containing metal objects in TACO.")

    images = annotations_data.get("images", [])
    metal_images = [img for img in images if img["id"] in metal_image_ids]

    if args.max_samples and len(metal_images) > args.max_samples:
        metal_images = metal_images[:args.max_samples]

    print(f"Targeting download of {len(metal_images)} metal images to {args.output_dir}...")

    downloaded_count = 0
    for idx, img in enumerate(metal_images):
        image_id = img["id"]
        flickr_url = img.get("flickr_url") or img.get("flickr_640_url")
        
        if not flickr_url:
            print(f"[{idx+1}/{len(metal_images)}] No URL for image ID {image_id}")
            continue

        output_path = args.output_dir / f"taco_metal_{image_id}.jpg"
        if output_path.exists():
            downloaded_count += 1
            continue

        print(f"[{idx+1}/{len(metal_images)}] Resolving {flickr_url}...")
        direct_url = get_direct_flickr_image_url(flickr_url)
        
        if not direct_url:
            print(f" -> Could not resolve direct image URL for Flickr page: {flickr_url}")
            continue
        
        print(f" -> Downloading from {direct_url} ...")
        success = download_image(direct_url, output_path)
        if success:
            downloaded_count += 1
            print(f" -> Saved to {output_path.name}")
            # // Flickr/Github 서버에 과도한 초당 요청(Requests per Second) 트래픽 부담을 주어 IP 차단(Blocking)을 당하지 않기 위해 다운로드 직후 0.5초 대기(sleep)합니다.
            time.sleep(0.5)
        else:
            print(" -> Download failed.")

    print(f"\nFinished! Downloaded/verified {downloaded_count} metal images from the TACO dataset.")

if __name__ == "__main__":
    main()
