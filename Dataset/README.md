#  쓰레기 분류 학습 데이터셋 (Garbage Classification Dataset)

본 폴더(`Dataset/`)는 모델 학습 및 검증에 사용되는 원본 이미지 데이터셋을 보관하는 위치입니다. 
데이터셋 전체 용량이 큰 관계로, 파일들은 GitHub 저장소에 직접 업로드하지 않고 클라우드 스토리지를 통해 공유합니다. (현재 `.gitignore`에 의해 이미지 파일들은 추적되지 않습니다.)

##  데이터셋 다운로드 및 세팅 방법

1. 아래의 구글 드라이브 링크에 접속하여 `dataset.zip` 파일을 다운로드합니다.
   * **다운로드 링크:** [https://drive.google.com/file/d/1L8TpC9F72u0hcoA-kqcD3gvvgZNQvrPn/view?usp=sharing]
2. 다운로드한 압축 파일을 이 `Dataset` 폴더 내부에 압축 해제합니다.
3. 압축 해제 후, 폴더 구조가 아래와 같이 구성되었는지 확인해 주세요.

## 📂 폴더 구조 안내

```text
Dataset/
├── .gitkeep
├── README.md
├── battery/
├── biological/
├── cardboard/
├── clothes/
├── glass/
├── metal/
├── non_garbage/        # Stage 1 (이진 분류) 음성 샘플
├── paper/
├── plastic/
├── shoes/
└── trash/
```

## non_garbage/ — Stage 1 음성 샘플

Stage 1(쓰레기 vs 비쓰레기 이진 분류) 학습용 "쓰레기 아님" 이미지.

- **출처**: [Intel Image Classification](https://huggingface.co/datasets/sfarrukhm/intel-image-classification) (Hugging Face)
- **장수**: 17,034장 (자연 풍경/건축물 6 카테고리: buildings, forest, glacier, mountain, sea, street)
- **파일명 규칙**: `intel_<split>_<category>_<index>.jpg`
- **License**: Intel 원본 데이터셋 라이선스 — 학습/연구 목적

`GarbageStage1Dataset`이 `non_garbage/`, `not_garbage/`, `non_waste/` 폴더의 이미지를 라벨 `0`으로, 10개 garbage 폴더의 이미지를 라벨 `1`로 자동 로드합니다.

---

## MS COCO 데이터셋을 활용한 일상 사물(비쓰레기) 데이터 보강

일상생활 속 다른 사물(자동차, 배, 가구, 책상 등)을 업로드했을 때 쓰레기로 오판하는 현상을 방지하기 위해, MS COCO 2017 데이터셋에서 관련 카테고리 이미지를 추가로 수집하여 `non_garbage/` 폴더에 구성합니다.

`fiftyone` 도구를 사용하여 특정 클래스 이미지만 골라 다운로드할 수 있습니다.

### 1. 도구 설치
```bash
pip install fiftyone
```

### 2. 수집 파이썬 스크립트 작성 및 실행
프로젝트 루트 폴더 혹은 임의의 위치에 아래 스크립트를 작성하여 실행하면, MS COCO 데이터셋에서 일반 사물 이미지를 1,000장 다운로드하여 `Dataset/non_garbage/` 폴더에 자동으로 배치합니다.

```python
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import shutil

# MS COCO-2017 데이터셋에서 원하는 클래스만 다운로드
print("MS COCO 데이터셋에서 일반 사물 이미지를 추출합니다...")
dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["validation"],  # validation 세트에서 추출 (수백~수천 장 규모에 적합)
    classes=["car", "boat", "chair", "dining table", "couch", "bed"], # 자동차, 보트, 의자, 식탁, 침상, 침대
    max_samples=1000
)

# 대상 폴더 생성 및 복사
target_dir = Path("Dataset/non_garbage")
target_dir.mkdir(parents=True, exist_ok=True)

print("이미지를 Dataset/non_garbage 폴더로 복사 중...")
for sample in dataset:
    source_path = Path(sample.filepath)
    # 파일명 중복을 방지하기 위해 고유 id를 파일명에 추가
    destination_path = target_dir / f"non_garbage_coco_{sample.id}.jpg"
    shutil.copy(source_path, destination_path)

print("완료!")
```