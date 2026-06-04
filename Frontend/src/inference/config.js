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

export const PREPROCESS = Object.freeze({
  resizeSize: 256,
  cropSize: 224,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
});

export const DEFAULT_MODEL_CONFIG = Object.freeze({
  stage1: '/models/stage1_improved.onnx',
  stage2: ['/models/stage2.onnx', '/models/stage2_seed123.onnx'],
  garbageThreshold: 0.8,
});

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

