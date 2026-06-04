import argparse
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# TACO dataset annotation URL
TACO_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/taco/master/data/annotations.json"

# Metal-related categories in TACO
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
    parser = argparse.ArgumentParser(description="Download Metal category images from the TACO dataset.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Dataset" / "metal")
    parser.add_argument("--max-samples", type=int, default=150, help="Maximum number of images to download")
    return parser.parse_args()

def get_direct_flickr_image_url(page_url):
    # If it is already a direct static link to a flickr image, return it directly
    if "staticflickr.com" in page_url or page_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return page_url

    try:
        req = urllib.request.Request(
            page_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        # Find og:image meta tag
        match = re.search(r'<meta\s+property="og:image"\s+content="([^"]+)"', html)
        if match:
            return match.group(1)
        
        # Fallback to search live.staticflickr.com or farm.staticflickr.com URLs
        match_static = re.search(r'https://[a-z0-9\.]+\.staticflickr\.com/[^"\']+', html)
        if match_static:
            return match_static.group(0)
            
    except Exception as e:
        print(f"Error resolving Flickr page {page_url}: {e}")
    return None

def download_image(url, output_path):
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
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading TACO annotations.json...")
    try:
        with urllib.request.urlopen(TACO_ANNOTATIONS_URL) as response:
            annotations_data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Failed to download annotations file: {e}")
        return

    # Map category ID to name, and filter for metal categories
    categories = annotations_data.get("categories", [])
    metal_category_ids = set()
    for cat in categories:
        if cat["name"] in METAL_CATEGORIES:
            metal_category_ids.add(cat["id"])
            print(f"Mapped TACO category: {cat['name']} (ID: {cat['id']})")

    # Find annotations belonging to metal categories
    annotations = annotations_data.get("annotations", [])
    metal_image_ids = set()
    for ann in annotations:
        if ann["category_id"] in metal_category_ids:
            metal_image_ids.add(ann["image_id"])

    print(f"Found {len(metal_image_ids)} images containing metal objects in TACO.")

    # Find image details
    images = annotations_data.get("images", [])
    metal_images = [img for img in images if img["id"] in metal_image_ids]

    # Limit samples
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
            # Skip if already exists
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
            # Respectful delay between requests
            time.sleep(0.5)
        else:
            print(" -> Download failed.")

    print(f"\nFinished! Downloaded/verified {downloaded_count} metal images from the TACO dataset.")

if __name__ == "__main__":
    main()
