from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CROP_SIZE = 224
# 학습/추론 전처리가 224x224 입력을 전제로 하므로 ONNX dummy input도 같은 크기로 고정한다.


def parse_args() -> argparse.Namespace:
    """ONNX export CLI 옵션을 파싱한다.

    Returns:
        argparse.Namespace: checkpoint 경로, output 경로, opset 버전.
    """

    parser = argparse.ArgumentParser(description="Export a trained PyTorch checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    """CLI 엔트리포인트로 PyTorch checkpoint를 ONNX 파일로 변환한다.

    Returns:
        None

    Raises:
        SystemExit: PyTorch 또는 torchvision 의존성이 없는 경우.
    """

    args = parse_args()
    try:
        # 서버나 샌드박스 환경에서는 홈 디렉터리 matplotlib 캐시 생성이 막힐 수 있어 임시 경로를 지정한다.
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

    # checkpoint에 저장된 모델명과 클래스 수를 기준으로 구조를 복원해야 classifier head 크기가 맞는다.
    # 학습 때 사용한 class_names 순서는 ONNX 추론의 출력 index 해석에도 그대로 사용된다.
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
    # dummy_input은 실제 값을 쓰지 않고 shape 추적에만 사용된다.
    # batch=1로 export하되 dynamic_axes를 지정해 런타임에서는 여러 장도 한 번에 넣을 수 있다.
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        # 배치 축만 동적으로 두면 CLI/브라우저에서 단일 이미지와 배치 추론을 같은 모델로 처리할 수 있다.
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
