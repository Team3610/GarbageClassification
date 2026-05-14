# 테스트 실행 및 검증

프로젝트의 AI 파이프라인과 프론트엔드 추론 로직을 검증한다.

## 실행 절차

1. 변경된 파일 확인 (`git diff develop...HEAD --name-only`)
2. 변경 영역에 따라 아래 해당 섹션을 실행

---

## AI 파이프라인 검증 (`AI/` 변경 시)

### 환경 확인
```bash
cd AI
python -c "import torch; print(torch.__version__)"
```

### 학습 코드 단위 검증
- 데이터로더가 정상 동작하는지 샘플 배치 1개 로드 테스트
- 모델 forward pass: 더미 입력(`torch.zeros(1, 3, 224, 224)`)으로 출력 shape 확인
- Stage 1 출력: `(batch, 2)` 또는 `(batch, 1)` (이진 분류)
- Stage 2 출력: `(batch, 10)` (10클래스)
- 앙상블 투표 함수: 동점 발생 케이스 포함 테스트

### ONNX 변환 검증
```bash
python AI/export/<변환스크립트>.py
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('path/to/model.onnx')
dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: dummy})
print('출력 shape:', out[0].shape)
"
```

### 정확도 스팟 체크
- 클래스별 대표 이미지 1장씩(총 10장)으로 Stage 2 추론 결과 확인
- Stage 1이 일반 사진(쓰레기 아님)을 올바르게 거부하는지 확인

---

## 프론트엔드 추론 검증 (`Frontend/` 변경 시)

### 전처리 일치 확인
- `Frontend/`의 JS 전처리(리사이즈·정규화 값)가 `AI/train/`의 transform과 동일한지 비교
- mean/std 값이 `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` (ImageNet 기본값) 기준으로 일치하는지 확인

### 브라우저 추론 흐름 점검
- ONNX 모델 로드 → 입력 텐서 생성 → 추론 → 결과 파싱 순서가 올바른지 코드 경로 추적
- `ort.Tensor` 생성 시 dtype (`float32`)과 shape (`[1, 3, 224, 224]`) 확인
- softmax 적용 후 argmax → 클래스 레이블 매핑이 학습 시 레이블 순서와 일치하는지

### 엣지 케이스 점검
- [ ] 이미지 없이 제출 시 에러 처리
- [ ] 지원하지 않는 파일 형식(PDF 등) 입력 시 처리
- [ ] Stage 1에서 쓰레기가 아닌 것으로 판단됐을 때 UI 분기 처리

---

## 출력 형식

```
## 테스트 범위
- 변경 파일: <목록>
- 실행한 검증 항목: <목록>

## 실패 항목
- [항목명] 실패 원인 및 재현 방법

## 경고
- 동작은 하지만 주의가 필요한 사항

## 통과 항목
- 정상 확인된 항목 목록

## 요약
전체 검증 결과 (1~2문장)
```

테스트 파일이 아직 없는 경우, 필요한 테스트 파일 목록과 작성 방법을 제안한다.
