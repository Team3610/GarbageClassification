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

## Stage 2 모델 학습

`train_stage2.py`는 10개 쓰레기 클래스를 분류하는 경량 전이학습 스크립트입니다.
기본 모델은 모바일/브라우저 추론을 고려해 `MobileNetV3-Small`을 사용합니다.
이미지 전처리는 `AI/preprocessing/transform.py`의 `train_transform`, `val_test_transform`을 재사용합니다.

```bash
python AI/train/train_stage2.py --epochs 10 --batch-size 32
```

EfficientNet-B0로 비교 실험을 하고 싶으면 아래처럼 실행합니다.

```bash
python AI/train/train_stage2.py --model-name efficientnet_b0 --epochs 10 --batch-size 32
```

설정과 출력 파일만 빠르게 확인하려면 일부 샘플만 사용해 smoke test를 실행할 수 있습니다.

```bash
python AI/train/train_stage2.py --epochs 1 --max-samples 100 --no-pretrained
```

학습 결과는 실행마다 `AI/train/runs/stage2/<model-name>-<timestamp>/` 아래에 저장됩니다.

```text
best_model.pt           # 검증 정확도가 가장 높은 체크포인트
metrics.csv             # epoch별 train/val loss, accuracy, macro F1
training_curves.png     # loss, accuracy 곡선
confusion_matrix.png    # 검증 데이터 혼동 행렬
config.json             # 학습 설정
summary.json            # 최고 성능 요약
```

기본값은 ImageNet 사전학습 가중치를 사용하는 전이학습입니다.
인터넷이 막혀 있거나 가중치 다운로드가 실패하면 `--no-pretrained`를 붙여 실행할 수 있습니다.

## ONNX 변환 및 계층적 추론

학습이 끝난 `best_model.pt`는 브라우저/ONNX Runtime 추론을 위해 ONNX로 변환합니다.

```bash
python AI/export/export_checkpoint_to_onnx.py \
  --checkpoint AI/train/runs/stage1/<run-name>/best_model.pt \
  --output Frontend/public/models/stage1.onnx

python AI/export/export_checkpoint_to_onnx.py \
  --checkpoint AI/train/runs/stage2/<run-name>/best_model.pt \
  --output Frontend/public/models/stage2.onnx
```

이미지 한 장에 대해 Stage 1(쓰레기 여부)과 Stage 2(10클래스 분류)를 순차 실행하려면 아래처럼 사용합니다.

```bash
python AI/inference/pipeline.py \
  --image sample.jpg \
  --stage1-model Frontend/public/models/stage1.onnx \
  --stage2-model Frontend/public/models/stage2.onnx
```

Stage 2 앙상블은 옵션입니다. `--stage2-model`을 여러 번 넘기면 각 모델의 확률을 평균내 최종 클래스를 고릅니다.

```bash
python AI/inference/pipeline.py \
  --image sample.jpg \
  --stage1-model Frontend/public/models/stage1.onnx \
  --stage2-model Frontend/public/models/stage2_a.onnx \
  --stage2-model Frontend/public/models/stage2_b.onnx
```

단일 모델과 앙상블의 정확도/추론 시간을 비교하려면 Stage 2 벤치마크를 실행합니다.

```bash
python AI/inference/benchmark_stage2.py --model Frontend/public/models/stage2.onnx

python AI/inference/benchmark_stage2.py \
  --model Frontend/public/models/stage2_a.onnx \
  --model Frontend/public/models/stage2_b.onnx
```

Python 코드에서 직접 호출할 수도 있습니다.

```python
from AI.inference import HierarchicalGarbageClassifier

classifier = HierarchicalGarbageClassifier(
    stage1_model_path="AI/models/stage1.onnx",
    stage2_model_paths="AI/models/stage2.onnx",
)

result = classifier.predict("Dataset/plastic/plastic_1.jpg")
print(result.to_dict())
```

Stage 2 앙상블을 코드에서 사용할 때는 `stage2_model_paths`에 여러 모델 경로를 리스트로 넘깁니다.

```python
from AI.inference import HierarchicalGarbageClassifier

classifier = HierarchicalGarbageClassifier(
    stage1_model_path="AI/models/stage1.onnx",
    stage2_model_paths=[
        "AI/models/stage2.onnx",
        "AI/models/stage2_seed123.onnx",
    ],
)

result = classifier.predict("Dataset/plastic/plastic_1.jpg")
print(result.to_dict())
```
