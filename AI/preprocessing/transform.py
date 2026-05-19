import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# 파라미터 상수
RESIZE_SIZE = 256
CROP_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# 이미지에 padding 더해서 1:1 정사각형으로 변형
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2) # 가로 여백
        vp = int((max_wh - h) / 2) # 세로 여백
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        
        # 빈 공간은 검은색(0)으로 채움
        return F.pad(image, padding, fill=0, padding_mode='constant')

# 학습용(Train) 데이터 전처리 (패딩 적용 -> 리사이즈 -> 증강)
train_transform = transforms.Compose([
    SquarePad(),                                     # 1. 원본 비율 유지하며 정사각형으로 여백 채우기
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),   # 2. 256x256 크기로 resize
    transforms.RandomCrop(CROP_SIZE),                # 3. 224 크기로 무작위 자르기 (증강)
    transforms.RandomHorizontalFlip(p=0.5),          # 4. 50% 확률로 좌우 반전 (증강)
    transforms.ToTensor(),                           # 5. 텐서 변환 및 픽셀값 스케일링 (0~1)
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD) # 6. 정규화
])

# 검증(Validation) 및 테스트(Test)용 데이터 전처리 (패딩 적용 -> 리사이즈 -> 정중앙 크롭)
val_test_transform = transforms.Compose([
    SquarePad(),                                     # 1. 원본 비율 유지하며 정사각형으로 여백 채우기
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),   # 2. 256x256 크기로 resize
    transforms.CenterCrop(CROP_SIZE),                # 3. 224 크기로 정중앙 자르기 (평가용)
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])