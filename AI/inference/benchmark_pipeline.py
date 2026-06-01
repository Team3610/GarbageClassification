from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for the hierarchical garbage classification ONNX pipeline."
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument("--stage1-model", type=Path, required=True, help="Stage 1 ONNX model path")
    parser.add_argument(
        "--stage2-model",
        type=Path,
        action="append",
        required=True,
        help="Stage 2 ONNX model path. Pass multiple times for ensemble.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--garbage-threshold", type=float, default=0.5)
    return parser.parse_args()


def get_file_size_mb(path: Path) -> float:
    if not path.is_file():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    args = parse_args()
    
    try:
        from sklearn.metrics import accuracy_score
        from AI.inference.pipeline import HierarchicalGarbageClassifier
        from AI.train.datasets import GarbageDataset
    except ImportError as exc:
        raise SystemExit(
            "Required packages are missing. Install with `pip install -r requirements.txt`.\n"
            f"Original error: {exc}"
        ) from exc

    # 모델 파일 크기 출력
    s1_size = get_file_size_mb(args.stage1_model)
    s2_sizes = [get_file_size_mb(p) for p in args.stage2_model]
    
    print("=" * 60)
    print("[SIZE] MODEL SIZE BENCHMARK")
    print("-" * 60)
    print(f"Stage 1 Model ({args.stage1_model.name}): {s1_size:.2f} MB (Target: <= 3MB)")
    if s1_size <= 3.0:
        print(" -> Stage 1 size target: PASSED [OK]")
    else:
        print(" -> Stage 1 size target: FAILED [FAIL]")
        
    for i, (p, size) in enumerate(zip(args.stage2_model, s2_sizes)):
        print(f"Stage 2 Model {i+1} ({p.name}): {size:.2f} MB (Target: <= 5MB)")
        if size <= 5.0:
            print(f" -> Stage 2 Model {i+1} size target: PASSED [OK]")
        else:
            print(f" -> Stage 2 Model {i+1} size target: FAILED [FAIL]")
    print("=" * 60)

    # 데이터셋 로드
    # Stage 1 평가는 전체 데이터(비쓰레기 포함)를 대상으로 수행
    try:
        dataset_s1 = GarbageDataset(args.dataset_dir, mode="stage1", return_path=True)
    except Exception as e:
        print(f"Warning: Could not load stage1 dataset ({e}). Benchmark will run on stage2 dataset.")
        dataset_s1 = None

    # Stage 2 평가는 쓰레기 카테고리만 대상으로 수행
    try:
        dataset_s2 = GarbageDataset(args.dataset_dir, mode="stage2", return_path=True)
    except Exception as e:
        print(f"Error: Could not load dataset from {args.dataset_dir} ({e})")
        return

    # 파이프라인 생성
    print("Initializing Hierarchical Garbage Classifier...")
    pipeline = HierarchicalGarbageClassifier(
        stage1_model_path=args.stage1_model,
        stage2_model_paths=args.stage2_model,
        garbage_threshold=args.garbage_threshold,
    )

    # 1. Stage 1 정확도 측정 (비쓰레기/쓰레기 분류)
    s1_accuracies = []
    if dataset_s1:
        samples_s1 = dataset_s1.samples[:args.max_samples] if args.max_samples else dataset_s1.samples
        print(f"\nEvaluating Stage 1 accuracy on {len(samples_s1)} samples...")
        
        s1_true = []
        s1_pred = []
        for path, label in samples_s1:
            # 예측값 얻기
            res = pipeline.predict(path)
            # True label: 0=NonGarbage, 1=Garbage
            s1_true.append(label)
            s1_pred.append(1 if res.stage1_label == "Garbage" else 0)
            
        s1_acc = accuracy_score(s1_true, s1_pred) * 100
        print(f"Stage 1 Accuracy: {s1_acc:.2f}% (Target: >= 90%)")
        if s1_acc >= 90.0:
            print(" -> Stage 1 accuracy target: PASSED [OK]")
        else:
            print(" -> Stage 1 accuracy target: FAILED [FAIL]")
    else:
        s1_acc = None

    # 2. Stage 2 정확도 측정 및 레이턴시 분석 (쓰레기 이미지들만 입력으로 보낸 경우의 Top-1 Accuracy)
    samples_s2 = dataset_s2.samples[:args.max_samples] if args.max_samples else dataset_s2.samples
    print(f"\nEvaluating Stage 2 and latency on {len(samples_s2)} samples...")
    
    s2_true = []
    s2_pred = []
    s1_latencies = []
    s2_latencies = []
    total_latencies = []
    
    for path, label in samples_s2:
        # pipeline.predict()
        res = pipeline.predict(path)
        
        # Stage 2 accuracy 평가는 Stage 1 결과와 상관없이 Stage 2 모델의 정확도 판단
        # 단, pipeline.predict()는 Stage 1이 non-garbage라 판단하면 Stage 2를 건너뛰므로, 
        # Stage 2의 순수한 Top-1 정확도를 측정하기 위해 Stage 2 분류 결과를 직접 확인합니다.
        # res.stage2_label이 None인 경우는 Stage 1에서 필터링된 것이므로, 
        # 강제로 stage2 모델들로 직접 예측을 구하여 측정해야 정확합니다.
        if res.stage2_label is not None:
            pred_class = res.stage2_label
        else:
            # 강제 Stage 2 예측 수행 (accuray 측정을 위함)
            # _predict_stage2_probabilities 사용
            from AI.inference.pipeline import load_image
            img = load_image(path)
            probs = pipeline._predict_stage2_probabilities(img)
            pred_idx = int(np.argmax(probs))
            from AI.inference.pipeline import GARBAGE_CLASSES
            pred_class = GARBAGE_CLASSES[pred_idx]
            
        s2_true.append(dataset_s2.class_to_idx[dataset_s2.samples[0][0].parent.name.capitalize()] if hasattr(dataset_s2, 'class_to_idx') else label)
        # dataset_s2.samples[i] 의 label 번호가 실제 정답
        s2_true[-1] = label
        
        # 예측 문자열을 인덱스로 변환
        from AI.train.datasets import CLASS_TO_IDX
        s2_pred.append(CLASS_TO_IDX[pred_class])
        
        s1_latencies.append(res.stage1_latency_ms)
        s2_latencies.append(res.stage2_latency_ms)
        total_latencies.append(res.total_latency_ms)

    s2_acc = accuracy_score(s2_true, s2_pred) * 100
    avg_s1_lat = np.mean(s1_latencies)
    avg_s2_lat = np.mean(s2_latencies)
    avg_total_lat = np.mean(total_latencies)
    p95_total_lat = np.percentile(total_latencies, 95)
    
    print(f"Stage 2 Top-1 Accuracy: {s2_acc:.2f}% (Target: >= 80%)")
    if s2_acc >= 80.0:
        print(" -> Stage 2 accuracy target: PASSED [OK]")
    else:
        print(" -> Stage 2 accuracy target: FAILED [FAIL]")
        
    print("-" * 60)
    print("[LATENCY] LATENCY BENCHMARK (ms)")
    print("-" * 60)
    print(f"Avg Stage 1 Latency  : {avg_s1_lat:.2f} ms")
    print(f"Avg Stage 2 Latency  : {avg_s2_lat:.2f} ms")
    print(f"Avg Total Latency    : {avg_total_lat:.2f} ms (Target: <= 500ms)")
    print(f"p95 Total Latency    : {p95_total_lat:.2f} ms")
    if avg_total_lat <= 500.0:
        print(" -> Latency target: PASSED [OK]")
    else:
        print(" -> Latency target: FAILED [FAIL]")
    print("=" * 60)


if __name__ == "__main__":
    main()
