# Speech Recognition (STT) Research

Date: 2026-02-22

## 1. Overview

Gesture control + gaze tracking 시스템에 음성 명령을 추가하기 위한 STT 모델 조사.
핵심 요구사항: **로컬 실행**, **한국어 지원**, **경량**, **Rust 배포 가능**.

## 2. STT Models Comparison (Korean Support)

### 한국어 성능 비교

| Model | Params | Size | Korean CER (Fleurs) | Korean CER (CV17) | ONNX | Rust 지원 | 최소 사양 |
|-------|--------|------|--------------------|--------------------|------|----------|----------|
| **Moonshine Tiny KO** | **27M** | **~26MB** | **8.9** | **14.94** | **Yes** | ort crate | 라즈베리파이급 |
| Whisper Tiny | 39M | 75MB | 15.83 | 37.27 | Yes | whisper-rs, candle, ort | RAM 1GB |
| Whisper Small | 244M | 500MB | ~8.0 (추정) | ~12.0 (추정) | Yes | whisper-rs, candle, ort | RAM 2GB |
| Whisper V3 Turbo | 809M | 1.6GB | 최고 | 최고 | Yes | whisper-rs | VRAM 6GB |
| Voxtral Mini 4B | 4B | ~8GB | 지원 | - | No | burn (순수 Rust) | GPU 필수 |

> CER = Character Error Rate (낮을수록 좋음). 한국어는 WER 대신 CER이 표준 지표.

### 핵심 발견: Moonshine Tiny KO

- **27M params로 Whisper Tiny(39M)보다 작으면서 CER이 절반** (8.9 vs 15.83)
- Whisper Medium(28x 크기)과 비슷한 성능
- 한국어 전용 특화 학습 (72,000시간 오디오)
- **ONNX 변환 모델 공개**: [onnx-community/moonshine-tiny-ko-ONNX](https://huggingface.co/onnx-community/moonshine-tiny-ko-ONNX)
- 제한사항: 환각(hallucination) 경향, 짧은 오디오에서 성능 저하

## 3. Rust STT Ecosystem

### Crate 비교

| Crate | 기반 | 모델 | GPU 지원 | 특징 |
|-------|------|------|---------|------|
| **[whisper-rs](https://crates.io/crates/whisper-rs)** | whisper.cpp (C++) | Whisper 전체 | CUDA, Metal, Vulkan | 가장 성숙, FFI 바인딩 |
| **[ort](https://github.com/pykeio/ort)** | ONNX Runtime | 모든 ONNX | CUDA, DirectML, TensorRT | 범용, Moonshine ONNX 사용 가능 |
| [candle](https://github.com/huggingface/candle) | HuggingFace | Whisper 등 | CUDA, Metal | 순수 Rust ML 프레임워크 |
| [whisper-burn](https://github.com/Gadersd/whisper-burn) | burn framework | Whisper | WGPU, TCH | 순수 Rust, 아직 불안정 |
| [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) | burn framework | Voxtral Mini 4B | GPU 필수 | 스트리밍 지원, 대형 모델 |
| [faster-whisper-rs](https://github.com/CodersCreative/faster-whisper-rs) | faster-whisper (Python) | Whisper | CUDA | Python FFI, CTranslate2 |

### 추천 조합

**Option A: Moonshine Tiny KO + ort (최경량)**
```
장점: 26MB 모델, ONNX 바로 사용, DirectML GPU 가속
단점: Moonshine의 ONNX encoder-decoder를 ort에서 직접 구동하려면 토크나이저+디코딩 루프 직접 구현 필요
크기: exe ~5MB + ONNX Runtime ~20MB + 모델 ~26MB = ~51MB
```

**Option B: Whisper Tiny + whisper-rs (가장 안정)**
```
장점: whisper.cpp 검증됨, 한줄 API, 스트리밍 지원
단점: 한국어 CER이 Moonshine보다 2배 나쁨 (15.83 vs 8.9)
크기: exe ~5MB + whisper.cpp ~2MB + 모델 75MB = ~82MB
```

**Option C: Whisper Small + whisper-rs (정확도 우선)**
```
장점: 한국어 CER ~8.0 수준, whisper.cpp 안정성
단점: 모델 500MB, RAM 2GB 필요
크기: exe ~5MB + whisper.cpp ~2MB + 모델 500MB = ~507MB
```

## 4. desktop-app-mic 프로젝트에서의 교훈

기존 `desktop-app-mic` 프로젝트 (Electron + Python STT)에서 얻은 인사이트:

### 아키텍처
- FasterWhisper (CTranslate2) + CUDA float16 → GPU에서 ~5-10초/30초청크
- CPU 폴백: int8 자동 전환
- 2분 세그먼트 완료 시 STT 트리거 (실시간 X, best-effort)

### 문제점 → 개선 방향
| desktop-app-mic 문제 | gesture-control 개선 |
|---------------------|---------------------|
| Python + PyInstaller = 150MB+ | Rust + ONNX = ~50MB |
| Electron 오버헤드 | 네이티브 바이너리 |
| CUDA 필수 (GPU 없으면 느림) | DirectML (모든 GPU) 또는 CPU |
| 30초 청크 → 2분 세그먼트로 변경 | 음성 명령이므로 실시간 필요 (VAD + 스트리밍) |

## 5. 음성 명령용 파이프라인 설계

### Gesture Control에서의 음성 역할

```
[시선] → 커서 위치 (어디를)
[손]   → 명령 종류 (클릭, 스크롤, 드래그)
[음성] → 복잡한 명령 ("이거 닫아", "새 탭 열어", "볼륨 올려")
```

### 실시간 음성 명령 파이프라인

```
마이크 입력 (16kHz, mono)
    ↓
VAD (Voice Activity Detection) — 음성 구간만 추출
    ↓ 음성 감지 시
STT (Moonshine Tiny KO 또는 Whisper)
    ↓ 텍스트
명령 매칭 (키워드 기반 또는 LLM)
    ↓
시스템 동작 실행
```

### VAD 옵션 (Rust)

| 라이브러리 | 크기 | 방식 | Rust 지원 |
|-----------|------|------|----------|
| Silero VAD | ~2MB | ONNX 모델 | ort crate |
| WebRTC VAD | <1MB | 규칙 기반 | webrtc-vad crate |
| 에너지 기반 | 0 | 볼륨 임계값 | 직접 구현 |

## 6. KVM 통합 고려사항

jet-agent의 kvm-ai-perception-research.md에서 이미 설계된 파이프라인:

```
[음성] Moonshine Tiny KO → 텍스트
[시선] MobileGaze → 화면 좌표
[제스처] MediaPipe/YOLO → 손 동작
    ↓
[LLM] 의도 파악 + STT 오류 보정
    ↓
[실행] jet-agent / KVM 명령
```

### 비용 구조 (기존 연구 결과)
| 방식 | 사용자당 월 비용 |
|------|---------------|
| 전부 클라우드 (Vision API) | $475,200 (불가능) |
| GPU 서빙 (Lightning) | $24.55 (적자) |
| **전부 로컬 + 텍스트 API만** | **$2.00 (흑자 73%)** |

## 7. 최종 추천

### 단기 (gesture-control 프로토타입)

**Moonshine Tiny KO + ort** (ONNX)
- 이유: 가장 작고(27M), 한국어 최적화, ONNX 공개
- Gaze tracking도 ort 사용 → 동일 런타임 공유
- 총 모델 크기: gaze(4.8MB) + face(2MB) + hand(8MB) + speech(26MB) = **~41MB**

### 장기 (KVM 통합)

**whisper-rs (Whisper Small)** + Moonshine KO 비교 후 결정
- KVM에서는 정확도가 더 중요 (명령 오인식 = 잘못된 PC 제어)
- Whisper Small(CER ~8.0)과 Moonshine KO(CER 8.9) 실사용 비교 필요
- GPU가 있으면 Whisper Small이 유리, 없으면 Moonshine이 유리

## 8. References

### Models
- [Moonshine Tiny KO](https://huggingface.co/UsefulSensors/moonshine-tiny-ko) — 27M params, Korean specialized
- [Moonshine Tiny KO ONNX](https://huggingface.co/onnx-community/moonshine-tiny-ko-ONNX) — ONNX converted
- [Moonshine v2 Paper](https://arxiv.org/html/2602.12241v1) — Streaming encoder ASR
- [Flavors of Moonshine Paper](https://arxiv.org/abs/2509.02523) — Edge device ASR
- [OpenAI Whisper](https://github.com/openai/whisper) — Original model
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — C/C++ port

### Rust Crates
- [whisper-rs](https://crates.io/crates/whisper-rs) — whisper.cpp Rust bindings
- [ort](https://github.com/pykeio/ort) — ONNX Runtime for Rust
- [candle](https://github.com/huggingface/candle) — HuggingFace Rust ML framework
- [whisper-burn](https://github.com/Gadersd/whisper-burn) — Whisper in burn framework
- [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) — Voxtral in Rust

### Related Project Docs
- `jet-agent/docs/kvm-ai-perception-research.md` — KVM AI perception 전체 설계
- `desktop-app-mic/specs/002-realtime-chunk-stt-gpu/research.md` — FasterWhisper GPU STT 구현 경험
- [STT Benchmarks 2026 (Northflank)](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
