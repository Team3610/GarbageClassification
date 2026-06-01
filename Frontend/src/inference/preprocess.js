import { PREPROCESS } from './config.js';

const { resizeSize, cropSize, mean, std } = PREPROCESS;

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
