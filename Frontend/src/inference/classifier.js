import * as ort from 'onnxruntime-web';

import { DEFAULT_MODEL_CONFIG, GARBAGE_CLASSES, PREPROCESS } from './config.js';
import { preprocessImage } from './preprocess.js';

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.common}/dist/`;
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

  async load(onProgress) {
    onProgress?.({ stage: 'sigmoid', loaded: 0, total: 1 });
    this.session = await ort.InferenceSession.create(this.config.modelPath, createSessionOptions());
    onProgress?.({ stage: 'sigmoid', loaded: 1, total: 1 });
  }

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
    
    // Apply sigmoid activation
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
