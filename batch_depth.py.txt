"""
레이블된 이미지에 대해 DepthAnything V2를 일괄 실행.
JSON이 있는 이미지만 처리, 이미 _depth.npy가 있으면 스킵.

Usage:
    python batch_depth.py [--cameras 1,2,3] [--scenarios a1_scenario_04]
                          [--overwrite] [--limit 10]
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ════════════════════════════════════════════
# Configuration — 여기만 수정하면 됩니다
# ════════════════════════════════════════════

_HERE = Path(__file__).parent

# 이미지 데이터 루트 폴더
DATA_ROOT = _HERE / "data"

# DepthAnything V2 ONNX 모델 경로
MODEL_PATH = _HERE / "models/models/depth_anything_v2_vit_l-r20240721/depth_anything_v2_vitl.onnx"

# depth 결과가 저장될 서브폴더명 (이미지 폴더 안에 생성됨)
DEPTH_DIR = "x-anylabeling-depth"

# 처리할 이미지 확장자
IMG_EXTENSIONS = ("*.jpg", )

# GPU 설정 (cuDNN 설치 후 USE_GPU = True로 변경)
USE_GPU = False
GPU_DEVICE_ID = 0  # GPU가 여러 장이면 0 또는 1

# ════════════════════════════════════════════


def load_model():
    print(f"Loading DepthAnything V2: {MODEL_PATH}")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if USE_GPU:
        providers = [("CUDAExecutionProvider", {"device_id": GPU_DEVICE_ID}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(MODEL_PATH), providers=providers, sess_options=opts)
    active = session.get_providers()[0]
    input_shape = session.get_inputs()[0].shape  # (1, 3, H, W)
    h, w = input_shape[2], input_shape[3]
    print(f"  provider: {active}, input: {h}x{w}")
    return session, h, w


def preprocess(image_bgr, model_h, model_w):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (model_w, model_h), interpolation=cv2.INTER_CUBIC)
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def run_depth(session, image_bgr, model_h, model_w):
    orig_h, orig_w = image_bgr.shape[:2]
    blob = preprocess(image_bgr, model_h, model_w)
    input_name = session.get_inputs()[0].name
    depth = session.run(None, {input_name: blob})[0]  # (1, H, W) or (1, 1, H, W)
    depth = depth.squeeze()
    depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth_norm.astype(np.float32)


def is_labeled(json_path):
    try:
        with open(json_path) as f:
            d = json.load(f)
        return bool(d.get("shapes"))
    except Exception:
        return False


def process_image(img_path, session, model_h, model_w, overwrite=False):
    stem = img_path.stem
    depth_out_dir = img_path.parent / DEPTH_DIR
    npy_path = depth_out_dir / f"{stem}_depth.npy"

    image = cv2.imread(str(img_path))
    if image is None:
        return "error:no_image"

    depth = run_depth(session, image, model_h, model_w)

    depth_out_dir.mkdir(exist_ok=True)
    np.save(str(npy_path), depth)
    return "ok"


def collect_images(scenario_dirs, cam_nums, overwrite):
    """처리할 이미지 경로 전체 수집."""
    images = []
    for scenario_dir in scenario_dirs:
        if not scenario_dir.is_dir():
            continue
        for cam_num in cam_nums:
            cam_dir = scenario_dir / f"Camera_{cam_num}"
            if not cam_dir.exists():
                continue
            for ext in IMG_EXTENSIONS:
                for img_path in sorted(cam_dir.glob(ext)):
                    json_path = img_path.with_suffix(".json")
                    if not (json_path.exists() and is_labeled(json_path)):
                        continue
                    npy_path = cam_dir / DEPTH_DIR / f"{img_path.stem}_depth.npy"
                    if overwrite or not npy_path.exists():
                        images.append(img_path)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", default="1,2,3,4,5")
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="병렬 스레드 수 (기본 1)")
    args = parser.parse_args()

    cam_nums = [c.strip() for c in args.cameras.split(",")]
    scenario_dirs = (
        [DATA_ROOT / s.strip() for s in args.scenarios.split(",")]
        if args.scenarios else sorted(DATA_ROOT.glob("*/"))
    )

    session, model_h, model_w = load_model()

    images = collect_images(scenario_dirs, cam_nums, args.overwrite)
    if args.limit:
        images = images[:args.limit]

    print(f"\n총 {len(images)}개 이미지 처리 (workers={args.workers})")

    total = skipped = errors = 0
    _print_lock = threading.Lock()

    def task(img_path):
        result = process_image(img_path, session, model_h, model_w, args.overwrite)
        with _print_lock:
            rel = img_path.relative_to(DATA_ROOT)
            print(f"  {rel} → {result}")
        return result

    if args.workers == 1:
        for img_path in images:
            result = task(img_path)
            total += 1
            if result == "skipped": skipped += 1
            elif result.startswith("error"): errors += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(task, p): p for p in images}
            for fut in as_completed(futures):
                total += 1
                result = fut.result()
                if result == "skipped": skipped += 1
                elif result.startswith("error"): errors += 1

    print(f"\n{'='*40}")
    print(f"완료. 처리={total}, 스킵={skipped}, 오류={errors}")


if __name__ == "__main__":
    main()
