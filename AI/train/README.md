# AI 학습 유틸리티

`AI/train/`은 PyTorch 학습에 필요한 데이터셋 준비 코드와 커스텀 Dataset 클래스를 관리합니다.

## 데이터셋 준비

먼저 프로젝트 루트에서 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

구글 드라이브의 `dataset.zip`을 자동으로 다운로드하고 `Dataset/` 폴더에 압축 해제하려면 아래 명령을 실행합니다.

```bash
python AI/train/prepare_dataset.py
```

이미 `dataset.zip`을 직접 다운로드한 경우에는 다운로드를 건너뛰고 압축 해제만 할 수 있습니다.

```bash
python AI/train/prepare_dataset.py --skip-download --zip-path Dataset/dataset.zip
```

압축 해제 후에는 아래 10개 클래스 폴더가 `Dataset/` 안에 있어야 합니다.

```text
Dataset/
├── battery/
├── biological/
├── cardboard/
├── clothes/
├── glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
└── trash/
```

## PyTorch Dataset 사용법

`datasets.py`에는 계층적 분류 파이프라인에 맞춘 Dataset 클래스가 들어 있습니다.

- `GarbageStage1Dataset`: Stage 1 이진 분류용
- `GarbageStage2Dataset`: Stage 2 10개 클래스 분류용
- `GarbageDataset`: `mode` 파라미터로 Stage 1/2를 직접 선택하는 기본 클래스

예시:

```python
from torch.utils.data import DataLoader
from torchvision import transforms

from AI.train import GarbageStage1Dataset, GarbageStage2Dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

stage1_dataset = GarbageStage1Dataset("Dataset", transform=transform)
stage2_dataset = GarbageStage2Dataset("Dataset", transform=transform)

stage1_loader = DataLoader(stage1_dataset, batch_size=32, shuffle=True)
stage2_loader = DataLoader(stage2_dataset, batch_size=32, shuffle=True)
```

## Stage 1 라벨

Stage 1은 입력 이미지가 쓰레기인지 아닌지 판단하는 이진 분류입니다.

```text
0 NonGarbage
1 Garbage
```

현재 데이터셋의 10개 클래스 폴더는 모두 쓰레기 이미지이므로 `1`로 라벨링됩니다.
나중에 비쓰레기 이미지가 필요하면 `Dataset/non_garbage/`, `Dataset/not_garbage/`, `Dataset/non_waste/` 중 하나에 넣으면 `0`으로 라벨링됩니다.

## Stage 2 라벨

Stage 2는 쓰레기 이미지를 10개 클래스로 분류합니다.

```text
0 Clothes
1 Glass
2 Plastic
3 Shoes
4 Cardboard
5 Paper
6 Metal
7 Battery
8 Biological
9 Trash
```

## 샘플 경로까지 확인하기

디버깅할 때 어떤 이미지가 어떤 라벨로 읽히는지 확인하려면 `return_path=True`를 사용할 수 있습니다.

```python
from AI.train import GarbageStage2Dataset

dataset = GarbageStage2Dataset("Dataset", return_path=True)
image, label, path = dataset[0]

print(label, path)
```
