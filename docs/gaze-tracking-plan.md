# Gaze Tracking Implementation Plan

## Current State
- `gaze.rs`: YuNet face detection + MobileGaze estimation 구현 완료 (컴파일 OK)
- Models: `face_detection.onnx` (YuNet 640x640 CHW), `gaze_mobileone_s0.onnx` (448x448 CHW)
- `main.rs`: gaze_tracker.detect() 호출 활성화, hand detection 주석 처리

## Phase 1: Face Detection 검증 (우선)
**목표**: YuNet 모델이 실제로 얼굴을 찾는지 확인

- [ ] Python으로 face_detection.onnx 출력 구조 검증 (cls/obj/bbox 디코딩)
- [ ] Rust에서 detect_face() 결과 로깅 (score, bbox 좌표)
- [ ] debug_view에서 face box가 정확히 그려지는지 확인
- [ ] 실패 시: YuNet output 파싱 로직 수정 (exp() 디코딩, grid 매핑 등)

**Apollo 가이드라인 적용**:
- `Result<T, E>` 전파 (`?` 연산자), `unwrap()` 금지
- 텐서 슬라이싱에 bounds check 추가

## Phase 2: Gaze Estimation 검증
**목표**: yaw/pitch 각도가 합리적인 값인지 확인

- [ ] Python으로 gaze_mobileone_s0.onnx 테스트 (face crop → yaw/pitch)
- [ ] bins_to_angle() 로직 검증 (90-bin → degree 변환)
- [ ] 좌/우 봤을 때 yaw 부호, 위/아래 봤을 때 pitch 부호 확인
- [ ] bin_width / offset 파라미터 조정 (모델별 다를 수 있음)

## Phase 3: Face Detection 개선
**목표**: 안정적 얼굴 추적

- [ ] 현재 stride-8 우선 → 전체 stride (8/16/32) NMS 통합
- [ ] OneEuroFilter로 face box 좌표 스무딩
- [ ] 얼굴 크기 기반 유효성 검사 (너무 작거나 큰 face box 제거)
- [ ] face tracking: 이전 프레임 face box 기반 search region 제한 (성능 최적화)

**Apollo 가이드라인 적용**:
- `&[f32]` borrowing (불필요한 clone 방지)
- Iterator 활용 (manual loop 대신 `.iter().enumerate().max_by()`)

## Phase 4: Screen Mapping
**목표**: yaw/pitch → 화면 좌표 변환

- [ ] 선형 매핑: `screen_x = 0.5 + (yaw / YAW_RANGE) * 0.5`
- [ ] YAW_RANGE / PITCH_RANGE 상수 튜닝 (기본값: 35도, 25도)
- [ ] OneEuroFilter로 커서 좌표 스무딩
- [ ] debug_view에 gaze point 표시 (십자 커서)

## Phase 5: Gaze + Hand Fusion (미래)
**목표**: 시선으로 대략적 위치, 손으로 정밀 제어

- [ ] 시선 = 커서 위치 (coarse), 손 제스처 = 클릭/드래그/스크롤
- [ ] AOI (Area of Interest) 방식: 시선 주변 영역 내에서 손으로 미세 조정
- [ ] hand detection 재활성화, 동시 추론 (face + hand)

## Model I/O Reference

### face_detection.onnx (YuNet)
```
IN:  input [1, 3, 640, 640] float32  (CHW, [0..1])
OUT: cls_8   [1, 6400, 1]   - classification scores (stride 8, 80x80 grid)
OUT: obj_8   [1, 6400, 1]   - objectness scores
OUT: bbox_8  [1, 6400, 4]   - bounding boxes (cx_offset, cy_offset, log_w, log_h)
OUT: kps_8   [1, 6400, 10]  - 5 facial keypoints (x,y pairs)
OUT: cls_16  [1, 1600, 1]   - stride 16 (40x40 grid)
OUT: obj_16  [1, 1600, 1]
OUT: bbox_16 [1, 1600, 4]
OUT: kps_16  [1, 1600, 10]
OUT: cls_32  [1, 400, 1]    - stride 32 (20x20 grid)
OUT: obj_32  [1, 400, 1]
OUT: bbox_32 [1, 400, 4]
OUT: kps_32  [1, 400, 10]
```

### gaze_mobileone_s0.onnx (MobileGaze L2CS)
```
IN:  input [1, 3, 448, 448] float32  (CHW, [0..1])
OUT: yaw   [1, 90]  - 90-bin classification (-99 to +99 degrees)
OUT: pitch [1, 90]  - 90-bin classification (-99 to +99 degrees)
```

## File Structure
```
src/gaze.rs      - GazeTracker (face detection + gaze estimation)
src/main.rs      - gaze_tracker.detect() in main loop
src/debug_view.rs - face box rendering (cyan rect)
src/controller.rs - GazeDirection → GazeSwitch events
```

## Known Risks
1. YuNet bbox decoding: `exp()` 기반이라 파라미터 오류 시 bbox 폭발
2. MobileGaze bin_width: 모델 학습 시 사용한 range와 불일치 가능
3. CHW vs HWC: face/gaze 모델은 CHW, hand 모델은 HWC — 혼동 주의
4. 조명/안경: 시선 추정 정확도에 큰 영향
