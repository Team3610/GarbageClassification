# CLAUDE.md — GarbageClassification

## 프로젝트 개요

쓰레기 이미지를 입력하면 분리수거 분류 결과를 알려주는 웹 서비스.
별도 API 서버 없이 브라우저 내에서 직접 추론(client-side inference)한다.

**팀 구성**

| 이름 | 역할 |
|------|------|
| 최재현 | PM, 프로젝트 관리 |
| 권시헌 | 데이터 전처리 |
| 김건민 | 모델 학습 |
| 김민성 | 웹 페이지 개발 |

---

## 아키텍처

### 추론 파이프라인 (계층적 앙상블)

```
입력 이미지
    │
    ▼
[Stage 1] 쓰레기 판별 모델 (binary classifier)
    │  쓰레기가 아님 → "쓰레기가 아닙니다" 반환
    │  쓰레기임 ↓
    ▼
[Stage 2] 분류 앙상블 (다수결 투표)
    ├── EfficientNet-Lite
    ├── MobileNet
    └── (추가 경량 모델)
    │
    ▼
최종 분류 결과 (10개 클래스)
```

### 분류 클래스 (10종)

| # | 클래스 | 설명 |
|---|--------|------|
| 1 | Clothes | 의류, 헌 옷, 직물류 |
| 2 | Glass | 유리병, 유리 조각 |
| 3 | Plastic | 페트병, 플라스틱 용기 |
| 4 | Shoes | 신발, 슬리퍼 |
| 5 | Cardboard | 골판지, 택배 상자 |
| 6 | Paper | 일반 종이, 신문지 |
| 7 | Metal | 알루미늄 캔, 고철 |
| 8 | Battery | 폐건전지, 배터리류 |
| 9 | Biological | 음식물 쓰레기, 나뭇잎 |
| 10 | Trash | 기타 일반 쓰레기 |

### 기술 스택

| 영역 | 기술 |
|------|------|
| 모델 학습 | PyTorch |
| 모델 경량화/변환 | ONNX → ONNX Runtime Web 또는 TensorFlow.js |
| 브라우저 추론 | ONNX Runtime Web (`ort`) |
| 프론트엔드 | (결정 전 — `Frontend/` 참고) |

### 폴더 구조

```
GarbageClassification/
├── AI/
│   ├── train/       # PyTorch 학습 코드
│   ├── inference/   # 추론 유틸리티
│   └── export/      # ONNX 변환 스크립트
├── Frontend/        # 웹 페이지 (브라우저 추론 포함)
├── Dataset/         # 데이터셋 (git-ignored)
├── Docs/            # 문서
└── Assets/          # 아이콘, 이미지 등 정적 자산
```

---

## Git 규칙

### 브랜치 전략 (Git Flow)

```
main          ← 배포 가능한 최종 상태
develop       ← 통합 브랜치
feature/<이름> ← 기능 개발
fix/<이름>     ← 버그 수정
release/<버전> ← 릴리즈 준비
hotfix/<이름>  ← 긴급 수정
```

- 모든 기능 개발은 `develop`에서 `feature/` 브랜치를 분기
- PR은 `feature/*` → `develop` 으로 생성
- `main` 직접 push 금지

### 커밋 메시지 형식

```
태그: 한국어 또는 영어 설명
```

| 태그 | 용도 |
|------|------|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `chore` | 설정, 빌드, 패키지 변경 |
| `docs` | 문서 수정 |
| `test` | 테스트 추가/수정 |
| `refactor` | 리팩토링 (기능 변화 없음) |
| `style` | 코드 포맷, 세미콜론 등 |

예시:
```
feat: EfficientNet-Lite 학습 스크립트 추가
fix: ONNX 변환 시 배치 차원 오류 수정
docs: 클래스 정의 문서 업데이트
```

- author/co-author에 Claude를 남기지 않는다.

### 이슈 & PR

- 작업 시작 전 이슈를 먼저 생성한다.
- PR 제목은 커밋 태그 형식을 따른다: `feat: 설명`
- PR 본문은 `.github/pull_request_template.md` 양식을 사용한다.
- PR은 관련 이슈를 `Closes #번호`로 연결한다.
- 코드 리뷰 없이 `develop` merge 금지.

---

## 개발 가이드

### 모델 학습 (`AI/train/`)

- 학습 프레임워크: PyTorch
- Stage 1 (쓰레기 판별): 이진 분류, 경량 모델 (MobileNetV3-Small 등)
- Stage 2 (10클래스 분류): EfficientNet-Lite, MobileNetV2/V3 앙상블
- 앙상블 방식: 각 모델의 예측 클래스를 다수결 투표(hard voting)
- 학습 완료 후 `AI/export/`의 스크립트로 ONNX 변환

### 브라우저 추론 (`Frontend/`)

- 모델 포맷: `.onnx`
- 런타임: ONNX Runtime Web (`ort`) — 별도 백엔드 서버 불필요
- 추론 흐름: 이미지 업로드 → 전처리 → Stage1 → Stage2 → 결과 표시
- 모델 파일은 `Assets/` 또는 `Frontend/` 내 적절한 위치에 배치

### 데이터셋

- `Dataset/` 폴더는 `.gitignore`에 포함 — 원본 데이터는 커밋하지 않는다.
- 데이터 출처 및 전처리 방법은 `Docs/`에 문서화한다.
