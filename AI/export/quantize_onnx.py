from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """
    /**
     * CLI 인자를 파싱합니다.
     * @returns {argparse.Namespace} 파싱된 명령행 인수 객체
     */
    """
    parser = argparse.ArgumentParser(description="Quantize an ONNX model using dynamic or static quantization.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input FP32 ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="Path to output INT8 quantized ONNX model")
    parser.add_argument(
        "--mode",
        choices=("dynamic", "static"),
        default="static",
        help="Quantization mode (dynamic or static). Static is recommended for CNN accuracy.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "Dataset", help="Dataset directory for static calibration")
    # // --num-calibration 기본값 50은 너무 크지도 작지도 않게 활성화(Activation) 분포 정보를 복원하며, static 양자화 과정 속도를 높이기 위한 적정 수치입니다.
    parser.add_argument("--num-calibration", type=int, default=50, help="Number of calibration images for static mode")
    return parser.parse_args()


def get_calibration_images(dataset_dir: Path, num_images: int) -> list[Path]:
    """
    /**
     * static 캘리브레이션에 사용할 이미지 경로 목록을 생성합니다.
     * @param {Path} dataset_dir - 데이터셋 루트 디렉토리 경로
     * @param {int} num_images - 추출할 이미지 최대 개수
     * @returns {list[Path]} 이미지 파일 경로 리스트
     */
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_images = []
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            all_images.append(path)
    
    if not all_images:
        return []
        
    # // 특정 카테고리에 편향되지 않고 데이터 전체 분포에서 이미지를 균일한 간격(step)으로 샘플링하기 위해 셔플을 제외하고 슬라이싱합니다.
    step = max(1, len(all_images) // num_images)
    return all_images[::step][:num_images]


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {args.input}")

    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static, CalibrationDataReader, QuantFormat
        from AI.inference.pipeline import load_image, preprocess_image
    except ImportError as exc:
        raise SystemExit(
            "onnx, onnxruntime, and scikit-learn are required. Install dependencies with `pip install -r requirements.txt`.\n"
            f"Original error: {exc}"
        ) from exc

    print(f"Quantizing {args.input.name} using {args.mode} mode...")
    
    if args.mode == "dynamic":
        try:
            quantize_dynamic(
                model_input=args.input,
                model_output=args.output,
                op_types_to_quantize=["MatMul", "Gemm"],
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=True,
            )
        except Exception as e:
            raise SystemExit(f"Dynamic quantization failed: {e}")
            
    else:  # static mode
        class ONNXCalibrationDataReader(CalibrationDataReader):
            """
            /**
             * static 양자화를 위해 입력 텐서 데이터를 로드하는 CalibrationDataReader 구현체입니다.
             * @param {list[Path]} image_paths - 캘리브레이션용 이미지 파일 경로 목록
             * @param {str} input_name - ONNX 모델의 입력 레이어명
             */
            """
            def __init__(self, image_paths: list[Path], input_name: str):
                self.image_paths = image_paths
                self.input_name = input_name
                self.data_iter = iter(image_paths)

            def get_next(self) -> dict | None:
                try:
                    path = next(self.data_iter)
                    img = load_image(path)
                    tensor = preprocess_image(img)  # Shape (1, 3, 224, 224)
                    return {self.input_name: tensor}
                except StopIteration:
                    return None

        session = ort.InferenceSession(str(args.input), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        
        calib_images = get_calibration_images(args.dataset_dir, args.num_calibration)
        if not calib_images:
            raise SystemExit(f"No calibration images found in dataset directory: {args.dataset_dir}")
            
        print(f"Using {len(calib_images)} images for static calibration...")
        data_reader = ONNXCalibrationDataReader(calib_images, input_name)
        
        try:
            quantize_static(
                model_input=args.input,
                model_output=args.output,
                calibration_data_reader=data_reader,
                # // QDQ 형식은 활성화와 가중치에 명시적 QuantizeLinear/DequantizeLinear 노드를 배치하여 ONNX Runtime Web에서 가장 최적화된 연산 효율을 보장합니다.
                quant_format=QuantFormat.QDQ, 
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=True,
            )
        except Exception as e:
            raise SystemExit(f"Static quantization failed: {e}")

    input_size = args.input.stat().st_size / (1024 * 1024)
    output_size = args.output.stat().st_size / (1024 * 1024)
    reduction = (1.0 - (output_size / input_size)) * 100.0

    print(f"\nQuantization completed successfully!")
    print(f"Original Model (FP32) : {input_size:.2f} MB")
    print(f"Quantized Model (INT8) : {output_size:.2f} MB")
    print(f"Size Reduction         : {reduction:.2f}%")
    
    # // 모바일/웹 서비스 로딩 성능 기준(Stage 1은 3MB 이하, Stage 2는 5MB 이하)을 통과하는지 검사합니다.
    is_stage1 = "stage1" in args.input.name.lower()
    is_stage2 = "stage2" in args.input.name.lower()
    
    if is_stage1:
        if output_size <= 3.0:
            print("[PASS] Stage 1 Target (<= 3MB) PASSED!")
        else:
            print("[FAIL] Stage 1 Target (<= 3MB) FAILED.")
    elif is_stage2:
        if output_size <= 5.0:
            print("[PASS] Stage 2 Target (<= 5MB) PASSED!")
        else:
            print("[FAIL] Stage 2 Target (<= 5MB) FAILED.")


if __name__ == "__main__":
    main()
