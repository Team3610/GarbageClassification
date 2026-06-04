from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# // CROP_SIZE = 224는 ImageNet 데이터셋으로 사전학습된 backbone 모델(MobileNet_V3, EfficientNet 등)의 표준 해상도 요구사항에 부합하기 위해 설정된 해상도입니다.
CROP_SIZE = 224


def parse_args() -> argparse.Namespace:
    """
    /**
     * CLI 인자를 파싱합니다.
     * @returns {argparse.Namespace} 파싱된 명령행 인수 객체
     */
    """
    parser = argparse.ArgumentParser(description="Export a trained PyTorch checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    # // opset=17은 ONNX Runtime Web에서 WebAssembly 및 WebGL 실행 환경을 통해 모델을 구동할 때 안정적인 하방 호환성과 최신 최적화 연산자를 지원받기 위한 선택입니다.
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    """
    /**
     * PyTorch 가중치(.pt) 모델을 ONNX 포맷으로 내보냅니다.
     * @returns {None}
     */
    """
    args = parse_args()
    try:
        matplotlib_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

        import torch

        from AI.train.trainer import build_model
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and torchvision are required. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    class_names = tuple(checkpoint["class_names"])

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, CROP_SIZE, CROP_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        # // 웹 프론트엔드 환경에서 다양한 배치 사이즈의 단일/다중 추론 요청을 유연하게 처리할 수 있도록 dynamic_axes를 활성화합니다.
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )
    print(f"Exported ONNX model to: {args.output.resolve()}")
    print(f"classes={class_names}")


if __name__ == "__main__":
    main()
