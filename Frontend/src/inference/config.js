export const GARBAGE_CLASSES = [
  'Clothes',
  'Glass',
  'Plastic',
  'Shoes',
  'Cardboard',
  'Paper',
  'Metal',
  'Battery',
  'Biological',
  'Trash',
];

export const STAGE1_CLASSES = ['NonGarbage', 'Garbage'];

// 학습/ONNX export 파이프라인과 동일한 입력 전처리 가정이다.
// resize -> center crop 이후 ImageNet mean/std로 정규화한 NCHW float32 텐서를 사용한다.
export const PREPROCESS = Object.freeze({
  resizeSize: 256,
  cropSize: 224,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
});

// Stage1은 "쓰레기 여부"를 먼저 엄격하게 걸러 오분류 안내를 줄이는 역할이다.
// 0.8은 현재 데모에서 비대상 이미지를 보수적으로 차단하기 위한 기본값이다.
export const DEFAULT_MODEL_CONFIG = Object.freeze({
  stage1: '/models/stage1_improved.onnx',
  stage2: ['/models/stage2.onnx', '/models/stage2_seed123.onnx'],
  garbageThreshold: 0.8,
});

// UI에서 모델 스왑을 지원하기 위한 프리셋 목록이다.
// 각 경로는 Vercel 배포 시 Frontend/public/models 아래에서 정적 파일로 서빙된다.
export const MODEL_PRESETS = Object.freeze({
  fp32Ensemble: {
    label: 'FP32 앙상블 (정확도 우선)',
    stage1: '/models/stage1_improved.onnx',
    stage2: ['/models/stage2.onnx', '/models/stage2_seed123.onnx'],
    garbageThreshold: 0.8,
  },
  fp32Single: {
    label: 'FP32 단일 (균형)',
    stage1: '/models/stage1_improved.onnx',
    stage2: ['/models/stage2.onnx'],
    garbageThreshold: 0.8,
  },
  sigmoid: {
    label: '시그모이드 모델 (멀티레이블 기반)',
    type: 'sigmoid',
    modelPath: '/models/stage2_sigmoid.onnx',
    garbageThreshold: 0.8,
  },
});
