import * as ort from 'onnxruntime-web';

import { DEFAULT_MODEL_CONFIG, GARBAGE_CLASSES, PREPROCESS } from './config.js';
import { preprocessImage } from './preprocess.js';

// wasm 파일은 번들에 포함하지 않고 CDN에서 받아 Vercel 배포 크기와 초기 설정 부담을 줄인다.
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.common}/dist/`;
// 여러 모델을 순차 실행하는 데모 앱이라 브라우저별 worker/thread 차이를 줄이는 안정성 우선 설정이다.
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

function createSessionOptions() {
  return {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  };
}

export class HierarchicalClassifier {
  constructor(config = DEFAULT_MODEL_CONFIG) {
    if (!config?.stage1 || !Array.isArray(config?.stage2) || config.stage2.length === 0) {
      throw new Error('classifier 설정에는 stage1 경로와 비어있지 않은 stage2 배열이 필요합니다.');
    }
    this.config = config;
    this.stage1Session = null;
    this.stage2Sessions = [];
  }

  /**
   * Stage1 이진 모델과 Stage2 다중분류 모델들을 순서대로 로드한다.
   *
   * @param {(progress: { stage: string, loaded: number, total: number }) => void} [onProgress]
   * @returns {Promise<void>}
   */
  async load(onProgress) {
    const total = 1 + this.config.stage2.length;
    let loaded = 0;
    const report = (stage) => {
      loaded += 1;
      onProgress?.({ stage, loaded, total });
    };

    this.stage1Session = await ort.InferenceSession.create(this.config.stage1, createSessionOptions());
    report('stage1');

    this.stage2Sessions = [];
    for (const path of this.config.stage2) {
      const session = await ort.InferenceSession.create(path, createSessionOptions());
      this.stage2Sessions.push(session);
      report('stage2');
    }
  }

  /**
   * 업로드 이미지를 Stage1/Stage2 파이프라인으로 분류한다.
   *
   * @param {Blob | string | HTMLImageElement} imageSource 브라우저에서 로드 가능한 이미지 입력
   * @returns {Promise<object>} UI에서 바로 표시할 수 있는 분류 결과와 latency 정보
   */
  async predict(imageSource) {
    if (!this.stage1Session || this.stage2Sessions.length === 0) {
      throw new Error('모델이 로드되지 않았습니다. load()를 먼저 호출하세요.');
    }

    const tensorData = await preprocessImage(imageSource);
    const tensor = new ort.Tensor('float32', tensorData, [1, 3, PREPROCESS.cropSize, PREPROCESS.cropSize]);

    const stage1Started = performance.now();
    const stage1Probs = await runSoftmax(this.stage1Session, tensor);
    const stage1LatencyMs = performance.now() - stage1Started;

    const garbageConfidence = stage1Probs[1];
    // Stage1에서 먼저 비대상 이미지를 차단해야 Stage2가 무조건 10개 쓰레기 클래스로 끼워 맞추는 상황을 줄일 수 있다.
    if (garbageConfidence < this.config.garbageThreshold) {
      return {
        isGarbage: false,
        label: '쓰레기가 아닙니다',
        confidence: stage1Probs[0],
        stage1Confidence: stage1Probs[0],
        stage1LatencyMs,
        stage2LatencyMs: 0,
        ensembleUsed: this.stage2Sessions.length > 1,
        stage2ModelCount: this.stage2Sessions.length,
      };
    }

    const stage2Started = performance.now();
    const stage2Probs = await ensembleProbs(this.stage2Sessions, tensor);
    const stage2LatencyMs = performance.now() - stage2Started;

    const idx = argmax(stage2Probs);
    return {
      isGarbage: true,
      label: GARBAGE_CLASSES[idx],
      confidence: stage2Probs[idx],
      stage1Confidence: garbageConfidence,
      stage1LatencyMs,
      stage2LatencyMs,
      ensembleUsed: this.stage2Sessions.length > 1,
      stage2ModelCount: this.stage2Sessions.length,
    };
  }
}


export class SigmoidClassifier {
  constructor(config) {
    if (!config?.modelPath) {
      throw new Error('Sigmoid classifier 설정에는 modelPath 경로가 필요합니다.');
    }
    this.config = config;
    this.session = null;
  }

  /**
   * 단일 sigmoid ONNX 모델을 로드한다.
   *
   * @param {(progress: { stage: string, loaded: number, total: number }) => void} [onProgress]
   * @returns {Promise<void>}
   */
  async load(onProgress) {
    onProgress?.({ stage: 'sigmoid', loaded: 0, total: 1 });
    this.session = await ort.InferenceSession.create(this.config.modelPath, createSessionOptions());
    onProgress?.({ stage: 'sigmoid', loaded: 1, total: 1 });
  }

  /**
   * sigmoid 출력값을 클래스별 독립 확률로 해석해 가장 높은 클래스를 선택한다.
   *
   * @param {Blob | string | HTMLImageElement} imageSource 브라우저에서 로드 가능한 이미지 입력
   * @returns {Promise<object>} UI에서 바로 표시할 수 있는 분류 결과와 latency 정보
   */
  async predict(imageSource) {
    if (!this.session) {
      throw new Error('모델이 로드되지 않았습니다. load()를 먼저 호출하세요.');
    }

    const tensorData = await preprocessImage(imageSource);
    const tensor = new ort.Tensor('float32', tensorData, [1, 3, PREPROCESS.cropSize, PREPROCESS.cropSize]);

    const started = performance.now();
    const feeds = { [this.session.inputNames[0]]: tensor };
    const output = await this.session.run(feeds);
    const logits = Array.from(output[this.session.outputNames[0]].data);
    
    // softmax 모델과 달리 각 클래스 로그릿을 독립 확률로 해석한다.
    const probs = logits.map((val) => 1 / (1 + Math.exp(-val)));
    const latencyMs = performance.now() - started;

    const idx = argmax(probs);
    const maxProb = probs[idx];
    const threshold = this.config.garbageThreshold ?? 0.5;

    if (maxProb < threshold) {
      return {
        isGarbage: false,
        label: '쓰레기가 아닙니다',
        confidence: 1 - maxProb,
        stage1Confidence: maxProb,
        stage1LatencyMs: latencyMs,
        stage2LatencyMs: 0,
        ensembleUsed: false,
        stage2ModelCount: 0,
        modelType: 'sigmoid',
        latencyMs: latencyMs,
      };
    } else {
      return {
        isGarbage: true,
        label: GARBAGE_CLASSES[idx],
        confidence: maxProb,
        stage1Confidence: maxProb,
        stage1LatencyMs: 0,
        stage2LatencyMs: latencyMs,
        ensembleUsed: false,
        stage2ModelCount: 1,
        modelType: 'sigmoid',
        latencyMs: latencyMs,
      };
    }
  }
}



async function runSoftmax(session, tensor) {
  const feeds = { [session.inputNames[0]]: tensor };
  const output = await session.run(feeds);
  const logits = Array.from(output[session.outputNames[0]].data);
  return softmax(logits);
}

async function ensembleProbs(sessions, tensor) {
  const all = [];
  for (const session of sessions) {
    all.push(await runSoftmax(session, tensor));
  }

  // 모델별 argmax를 투표하지 않고 확률을 평균해 confidence와 UI 표시값을 함께 유지한다.
  const length = all[0].length;
  const avg = new Array(length).fill(0);
  for (const probs of all) {
    for (let i = 0; i < length; i++) avg[i] += probs[i];
  }
  return avg.map((value) => value / all.length);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((value) => value / sum);
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}
