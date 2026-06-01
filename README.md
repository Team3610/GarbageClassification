# GarbageClassification
쓰레기 분류기 - 이미지 기반 분리수거 분류 도움 서비스

## 🧑‍💻 소개

<br>

>👉프로젝트 요약  
본 프로젝트는 쓰레기 이미지 기반으로 종류를 분류하여 분리수거를 돕는 딥러닝 프로젝트입니다

<br>

## 🌏 팀원

| 조원 | 담당 역할 | 역할 |
| --- |:---: | :---: |
| 최재현 | PM  | 프로젝트 관리 |
| 권시헌 | 개발 | 데이터 전처리 |
| 김건민 | 개발 | 모델 학습 |
| 김민성 | 개발 | 웹 페이지 개발 |

## 프론트엔드 실행 방법

프론트엔드는 `Frontend` 디렉터리의 React + Vite 앱으로 구성되어 있습니다.

```bash
cd Frontend
npm install
npm run dev
```

실행 후 브라우저에서 `http://localhost:5173/`로 접속하면 화면을 확인할 수 있습니다.

프로덕션 빌드는 아래 명령어로 확인할 수 있습니다.

```bash
npm run build
```

## Vercel 배포 설정

Vercel에서는 프론트엔드 앱만 빌드하도록 아래 값으로 프로젝트를 설정합니다.

| 항목 | 값 |
| --- | --- |
| Root Directory | `Frontend` |
| Build Command | `npm run build` |
| Output Directory | `dist` |

ONNX 모델 파일은 `Frontend/public/models`에 배치되어 있으며, Vite 빌드 후 정적 파일로 제공됩니다.

| 모델 | 배포 후 접근 경로 |
| --- | --- |
| Stage1 쓰레기 판별 모델 | `/models/stage1.onnx` |
| Stage2 기본 분류 모델 | `/models/stage2.onnx` |
| Stage2 앙상블 모델 | `/models/stage2_seed123.onnx` |

배포 전 로컬에서 아래 명령으로 빌드가 통과하는지 확인합니다.

```bash
cd Frontend
npm install
npm run build
```

Vercel 계정 연결과 실제 배포는 Vercel 대시보드에서 직접 진행합니다. GitHub 저장소를 연결한 뒤 위 설정값을 입력하면 됩니다.

## v1.0.0 릴리즈 범위

v1.0.0에는 아래 기능을 포함합니다.

- Stage1/Stage2 ONNX 추론 파이프라인
- Python 추론 코드와 브라우저 추론 코드
- FP32 Stage2 앙상블 기본 구조
- FP32 단일 모델과 FP32 앙상블을 전환하는 모델 스왑 옵션
- 이미지 업로드 기반 분리수거 가이드 UI

## v1.0.0 릴리즈 절차

실제 Git 태그 생성, GitHub Release 생성, 원격 push는 자동으로 수행하지 않습니다. 릴리즈 담당자가 아래 절차를 직접 실행합니다.

```bash
git status
cd Frontend
npm install
npm run build
cd ..
```

변경사항 확인 후 커밋합니다.

```bash
git add README.md Frontend/package.json Frontend/package-lock.json Frontend/public/models
git commit -m "chore: v1.0.0 배포 설정 정리"
```

태그와 원격 반영은 릴리즈 담당자가 최종 확인 후 직접 실행합니다.

```bash
git tag v1.0.0
git push origin develop
git push origin v1.0.0
```

GitHub Release는 GitHub 웹 UI에서 `v1.0.0` 태그를 선택해 생성합니다. 릴리즈 노트에는 위 v1.0.0 포함 범위와 Vercel 배포 URL을 함께 적습니다.
