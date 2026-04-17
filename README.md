# X-AnyLabeling Toolkit

주행 영상 어노테이션 파이프라인을 위한 보조 툴 모음입니다.  
[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 기반으로, 레이블 현황 확인 / Depth 일괄 추출 / 결과 뷰어 기능을 제공합니다.

---

## 툴 구성

| 파일 | 설명 |
|---|---|
| `check_status.py` | 시나리오·카메라별 어노테이션 현황 실시간 모니터링 |
| `batch_depth.py` | 레이블된 이미지에 DepthAnything V2 일괄 실행 |
| `viewer/main.py` | 어노테이션 + Depth 결과 뷰어 (PyQt6 GUI) |

---

## 요구 사항

```
Python 3.10+
opencv-python
numpy
onnxruntime          # GPU 사용 시: onnxruntime-gpu
PyQt6                # viewer 전용
```

```bash
pip install opencv-python numpy onnxruntime PyQt6
```

---

## 데이터 구조

```
data/
└── <scenario>/
    └── Camera_<N>/
        ├── image_001.jpg
        ├── image_001.json       ← X-AnyLabeling 레이블
        └── x-anylabeling-depth/
            └── image_001_depth.npy   ← batch_depth.py 결과
```

---

## 사용법

### 1. check_status.py — 어노테이션 현황 확인

```bash
python check_status.py
```

- 3초마다 자동 갱신 (터미널 클리어 후 재출력)
- 수동 / 자동(autolabel) / depth 파일 수와 전체 이미지 수를 표시
- 아직 레이블이 없는 카메라에는 `← 필요` 표시

출력 예시:
```
시나리오             카메라          수동   자동  depth  전체이미지
--------------------------------------------------------------
a1_scenario_04      Camera_1           5     12     17          30
a1_scenario_04      Camera_2           0      0      0          30  ← 필요
```

---

### 2. batch_depth.py — Depth 일괄 추출

```bash
# 전체 시나리오, 전체 카메라
python batch_depth.py

# 특정 카메라만
python batch_depth.py --cameras 1,2,3

# 특정 시나리오만
python batch_depth.py --scenarios a1_scenario_04,a1_scenario_05

# 이미 처리된 파일도 덮어쓰기
python batch_depth.py --overwrite

# 테스트용 (10개만)
python batch_depth.py --limit 10

# 병렬 처리 (스레드 4개)
python batch_depth.py --workers 4
```

- JSON 레이블이 있는 이미지만 처리
- `_depth.npy`가 이미 존재하면 스킵 (--overwrite로 강제 재처리)
- 결과는 `Camera_N/x-anylabeling-depth/` 폴더에 저장

**GPU 사용 시** `batch_depth.py` 상단 설정 변경:
```python
USE_GPU = True
GPU_DEVICE_ID = 0
```

---

### 3. viewer — 어노테이션 뷰어

```bash
cd viewer
python main.py
```

- 이미지 + 어노테이션 오버레이 시각화
- Depth 결과(.npy) 함께 확인 가능
- `viewer/config.yaml`에서 데이터 경로 설정

---

## 모델

DepthAnything V2 ONNX 모델을 `models/` 폴더에 배치:

```
models/
└── models/
    └── depth_anything_v2_vit_l-r20240721/
        └── depth_anything_v2_vitl.onnx
```

> X-AnyLabeling 실행 시 자동 다운로드되거나, X-AnyLabeling 공식 모델 저장소에서 받을 수 있습니다.

---

## 빠른 실행

```bash
# 1. 패키지 설치
pip install opencv-python numpy onnxruntime PyQt6

# 2. 현황 확인 (3초 자동 갱신)
python check_status.py

# 3. Depth 일괄 추출
python batch_depth.py

# 4. 뷰어 실행
cd viewer && python main.py
```
