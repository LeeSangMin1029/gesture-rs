# Python Prototype Reference

Python 프로토타입(hand_tracker.py, mouse_controller.py)의 핵심 로직 요약.
Rust 포팅 시 참조용으로 보관.

## Gesture Classification (hand_tracker.py)

### 우선순위 (높은 것부터)
1. **FIST** — 4 fingers curled + thumb tucked → 모든 동작 차단
2. **THUMBS_UP** — 4 fingers curled + thumb 돌출 → 스크롤 원점 설정
3. **THUMBS_DOWN** — 4 fingers curled + thumb 아래 방향 → 최소화
4. **TWO_FINGERS** — index + middle만 펴짐 → 스크롤
5. **PINCH** — thumb+index 거리 < 25px → 클릭/드래그
6. **OPEN_HAND** — 4+ fingers → 포인터 이동
7. **POINTER** — 2-3 fingers (fallback) → 포인터 이동

### Thumb Extended 판정 (3가지 투표, 2/3 이상)
- X-direction: Right hand → tip.x > ip.x
- Straightness: CMC→MCP와 MCP→TIP 각도 < 50°
- Protrusion: tip-to-palm-center / hand-size > 0.45

### Thumbs Up 판정 (3가지 기준 모두 충족)
- thumb-to-fist-cluster / hand-size >= 0.50
- thumb-to-index / hand-size >= 0.30
- thumb 직선도 각도 < 60°

### Debounce
- 5프레임 버퍼 + Counter.most_common
- THUMBS_UP 감지 시 8프레임 scroll lock (FIST 흔들림 방지)

## Mouse Controller (mouse_controller.py)

### OneEuroFilter
- min_cutoff=0.8, beta=0.01 (포인터)
- min_cutoff=0.3, beta=0.005 (스크롤 — 더 무거운 스무딩)

### 포인터 이동
- 추적점: landmark[5] (index MCP)
- POINTER_GAIN = 2.5, POINTER_DEADZONE = 2px
- pyautogui.moveRel(dx * gain, dy * gain)

### 앵커 모드 (두 손)
- 왼손 FIST = 앵커 원점 설정
- 오른손 거리 = 마우스 속도
- ANCHOR_GAIN = 4.0, ANCHOR_DEADZONE = 15px

### 클릭/드래그
- index+thumb pinch < 16px → 클릭
- 0.4초 이상 유지 → 드래그 시작
- middle+thumb pinch < 16px → 더블 클릭

### Thumbs-up 스크롤
- 0.5초 정지 대기(lock-in) → origin Y 확정
- dy = current - origin → scroll(dy * 135 / 100)
- SCROLL_DEADZONE = 3px

### Two-finger 스크롤
- 추적점: landmark[9] (middle MCP)
- SCROLL_SENS = 135, SCROLL_DEADZONE = 3px

## 주요 상수
| 상수 | 값 | 용도 |
|---|---|---|
| PINCH_THRESHOLD | 25 (hand) / 16 (mouse) | 핀치 감지 거리 |
| POINTER_GAIN | 2.5 | 포인터 이동 배율 |
| POINTER_DEADZONE | 2 | 포인터 최소 이동 |
| ANCHOR_GAIN | 4.0 | 앵커 모드 이동 배율 |
| SCROLL_SENS | 135 | 스크롤 감도 |
| DRAG_HOLD_TIME | 0.4s | 드래그 전환 시간 |
