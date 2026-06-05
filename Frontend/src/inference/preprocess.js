import { PREPROCESS } from './config.js';

const { resizeSize, cropSize, mean, std } = PREPROCESS;

/**
 * 브라우저에서 선택한 이미지를 ONNX 모델 입력 텐서로 변환한다.
 *
 * @param {Blob | string | HTMLImageElement} imageSource 업로드 파일, 이미지 URL, 또는 로드된 이미지 엘리먼트
 * @returns {Promise<Float32Array>} [1, 3, cropSize, cropSize]에 들어갈 CHW 순서 float32 데이터
 */
export async function preprocessImage(imageSource) {
  const image = await loadImageElement(imageSource);
  const canvas = drawSquarePaddedResize(image, resizeSize);
  const pixels = centerCropImageData(canvas, cropSize);
  return normalizeToCHW(pixels);
}

function loadImageElement(source) {
  if (source instanceof HTMLImageElement && source.complete && source.naturalWidth > 0) {
    return Promise.resolve(source);
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('이미지를 불러오지 못했습니다.'));

    if (source instanceof Blob) {
      img.src = URL.createObjectURL(source);
    } else if (typeof source === 'string') {
      img.src = source;
    } else {
      reject(new Error('지원하지 않는 이미지 입력입니다.'));
    }
  });
}

function drawSquarePaddedResize(image, targetSize) {
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d');

  // 학습 시 사용한 letterbox 계열 입력을 맞추기 위해 원본 비율을 유지하고 남는 영역은 검정으로 채운다.
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, targetSize, targetSize);

  const scale = targetSize / Math.max(image.naturalWidth, image.naturalHeight);
  const drawW = Math.round(image.naturalWidth * scale);
  const drawH = Math.round(image.naturalHeight * scale);
  const offsetX = Math.floor((targetSize - drawW) / 2);
  const offsetY = Math.floor((targetSize - drawH) / 2);
  ctx.drawImage(image, offsetX, offsetY, drawW, drawH);
  return canvas;
}

function centerCropImageData(canvas, crop) {
  const ctx = canvas.getContext('2d');
  const left = Math.floor((canvas.width - crop) / 2);
  const top = Math.floor((canvas.height - crop) / 2);
  return ctx.getImageData(left, top, crop, crop).data;
}

function normalizeToCHW(rgba) {
  const planeSize = cropSize * cropSize;
  const tensor = new Float32Array(3 * planeSize);

  // ONNX 모델은 PyTorch export 기준의 NCHW 입력을 기대하므로 RGBA 픽셀을 채널별 plane으로 재배치한다.
  for (let i = 0; i < planeSize; i++) {
    const r = rgba[i * 4] / 255;
    const g = rgba[i * 4 + 1] / 255;
    const b = rgba[i * 4 + 2] / 255;
    tensor[i] = (r - mean[0]) / std[0];
    tensor[planeSize + i] = (g - mean[1]) / std[1];
    tensor[planeSize * 2 + i] = (b - mean[2]) / std[2];
  }
  return tensor;
}
