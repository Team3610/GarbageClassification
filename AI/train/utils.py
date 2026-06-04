import random


def seed_everything(seed: int) -> None:
    """데이터 분할과 모델 초기화를 같은 seed 기준으로 맞춰 실험 재현성을 높인다.

    Args:
        seed (int): Python, NumPy, PyTorch에 적용할 난수 seed.

    Returns:
        None
    """

    import numpy as np
    import torch

    # DataLoader worker나 일부 GPU 연산까지 완전 결정적으로 만들지는 못하지만,
    # 현재 학습 스크립트의 subset 추출, split, 모델 초기화 재현에는 충분한 기준점이 된다.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str):
    """auto 옵션에서 CUDA, Apple MPS, CPU 순서로 사용 가능한 학습 장치를 선택한다.

    Args:
        device_name (str): "auto", "cpu", "cuda", "mps" 중 하나.

    Returns:
        torch.device: 학습에 사용할 PyTorch device 객체.
    """

    import torch

    if device_name == "auto":
        # CUDA가 있으면 대체로 가장 빠르지만, Mac 개발 환경에서는 MPS가 CPU보다 유리해서 두 번째로 둔다.
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)
