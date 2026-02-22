# YOLO26 손 제스처 인식 리서치

> 조사일: 2026-02-22

## 1. Ultralytics YOLO26 개요

- **출시**: 2025년 9월, YOLO Vision 2025 (London)
- **공식 레포**: https://github.com/ultralytics/ultralytics
- **지원 태스크**: Detection, Segmentation, Pose Estimation, OBB, Classification

### 핵심 아키텍처 개선

| 항목 | 설명 |
|------|------|
| NMS 제거 | end-to-end 예측, 후처리 병목 제거 |
| DFL 모듈 제거 | 바운딩 박스 회귀를 경량 방식으로 대체 |
| MuSGD 옵티마이저 | SGD + Muon 하이브리드, 빠른 수렴 |
| 엣지 최적화 | GPU 없는 환경을 명시적 타겟 |

### CPU 성능 벤치마크

| 모델 | CPU 추론 | GPU 추론 | mAP | 비고 |
|------|---------|---------|-----|------|
| YOLO26n | 38.9ms | 1.7ms | 40.9% | **YOLO11n 대비 CPU 43% 빠름** |
| YOLO26s | 87.2ms | - | 47.2% | |
| YOLO26m | 220.0ms | - | 51.5% | |
| YOLO26l | 286.2ms | - | 53.0~53.4% | |

---

## 2. 손 제스처 인식 가능 여부

### 기본 제공 모델

- 기본 pose 모델(`yolo26n-pose.pt`)은 **COCO 17 keypoints (신체 관절)** 로 사전 훈련
- **손 제스처 사전 훈련 모델은 없음** → 커스텀 훈련 필수

### Hand Keypoints 데이터셋 (Ultralytics 공식)

- **이미지 수**: 26,768장 (훈련 18,776 / 검증 7,992)
- **클래스**: 1개 (Hand)
- **키포인트**: 21개 (손목 1 + 손가락당 4개)
- **설정 파일**: `hand-keypoints.yaml` (자동 다운로드)
- **문서**: https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/pose/hand-keypoints.md

### 커뮤니티 모델

- [yolov8x 손 제스처 파인튜닝](https://huggingface.co/lewiswatson/yolov8x-tuned-hand-gestures)
- [YOLO11n-pose 손 키포인트](https://github.com/chrismuntean/YOLO11n-pose-hands)

---

## 3. 눈 추적 / 얼굴 인식

| 기능 | YOLO26 지원 | 대안 |
|------|-----------|------|
| 눈 움직임 추적 | **미지원** | MediaPipe FaceMesh (478 landmarks) |
| 얼굴 감지 | 커스텀 훈련 필요 | WIDER FACE 데이터셋 |
| 얼굴 인식 (누구인지) | **미지원** (YOLO 영역 아님) | InsightFace, DeepFace, ArcFace |

---

## 4. 훈련 방법

### 환경 설정

```bash
python -m venv yolo-hand
source yolo-hand/Scripts/activate  # Windows Git Bash
pip install ultralytics
```

### 훈련 코드

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
results = model.train(
    data="hand-keypoints.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",        # GPU: device=0
    project="hand-gesture",
    name="yolo26n-hand"
)
```

### 훈련 시간 예상

| 환경 | 100 epochs |
|------|-----------|
| CPU만 | 10~24시간 |
| GTX 3060 | 2~4시간 |
| RTX 4090 | 30분~1시간 |

### 검증

```python
model = YOLO("hand-gesture/yolo26n-hand/weights/best.pt")
metrics = model.val()
print(f"mAP50: {metrics.pose.map50}")
```

---

## 5. 제스처 분류 로직

YOLO26 pose는 키포인트 좌표만 출력. 제스처 판별은 별도 로직 필요:

```python
def classify_gesture(keypoints):
    """21개 키포인트 → 제스처 분류"""
    # 손가락 끝: 엄지(4), 검지(8), 중지(12), 약지(16), 새끼(20)
    # 손가락 중간 관절: 엄지(3), 검지(6), 중지(10), 약지(14), 새끼(18)
    finger_tips = [4, 8, 12, 16, 20]
    finger_mids = [3, 6, 10, 14, 18]

    fingers_up = []
    for tip, mid in zip(finger_tips, finger_mids):
        fingers_up.append(keypoints[tip][1] < keypoints[mid][1])

    total = sum(fingers_up)
    if total == 0:
        return "fist"
    elif total == 5:
        return "open_hand"
    elif fingers_up == [False, True, True, False, False]:
        return "peace"
    elif fingers_up == [True, True, False, False, False]:
        return "point"
    elif fingers_up == [True, False, False, False, True]:
        return "rock"
    else:
        return f"fingers_{total}"
```

---

## 6. 배포 방법

### 모델 내보내기

```python
model = YOLO("hand-gesture/yolo26n-hand/weights/best.pt")

model.export(format="onnx")       # 범용
model.export(format="openvino")   # Intel CPU 최적화
model.export(format="tflite")     # 모바일/엣지
model.export(format="coreml")     # iOS/Mac
model.export(format="engine")     # NVIDIA TensorRT
```

### 배포 포맷 선택

| 배포 대상 | 포맷 | 특징 |
|----------|------|------|
| Windows 데스크탑 | `onnx` | 범용, 설치 쉬움 |
| Intel CPU 서버 | `openvino` | Intel CPU 2배 빠름 |
| NVIDIA GPU 서버 | `engine` (TensorRT) | 최고 속도 |
| Android | `tflite` | 경량, 모바일 최적화 |
| iOS | `coreml` | Apple 생태계 최적화 |
| 라즈베리파이 | `tflite` + INT8 양자화 | 초경량 |

### 실시간 추론 예시 (ONNX)

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.onnx")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        if r.keypoints is not None:
            for kpts in r.keypoints.data:
                gesture = classify_gesture(kpts.cpu().numpy())
                cv2.putText(frame, gesture, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 7. MediaPipe vs YOLO26 비교 요약

| 항목 | YOLO26 | MediaPipe |
|------|--------|-----------|
| 손 키포인트 | 21개 (훈련 후) | 21개 (즉시 사용) |
| 훈련 필요 | O (수 시간) | X |
| CPU 속도 | 38.9ms (26n) | ~33ms |
| GPU 속도 | 1.7ms | 보통 |
| 눈 추적 | 미지원 | O (FaceMesh) |
| 얼굴 인식 | 미지원 | 부분 지원 |
| 멀티 태스크 | 객체 감지 동시 가능 | 손/얼굴 별도 |
| 배포 유연성 | ONNX/TRT/TFLite 등 | TFLite/웹 |

---

## 참고 자료

- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [YOLO26 논문](https://arxiv.org/html/2509.25164v4)
- [YOLO26 Roboflow 가이드](https://blog.roboflow.com/yolo26/)
- [YOLO26 엣지 배포 가이드](https://neuralnet.solutions/yolo26-on-edge-devices-a-complete-guide)
- [Hand Keypoints 데이터셋 문서](https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/pose/hand-keypoints.md)
- [YOLO26 튜토리얼](https://medium.com/@zainshariff6506/a-simple-yolo26-tutorial-from-beginners-to-experts-b5aa491b8ace)
- [YOLO vs MediaPipe 비교](https://learnopencv.com/yolov7-pose-vs-mediapipe-in-human-pose-estimation/)
- [손 제스처 벤치마킹 논문](https://www.nature.com/articles/s41598-025-23925-9)
