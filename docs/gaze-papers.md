# Gaze Tracking Papers (2020-2025, accuracy순)

| # | Model | Year | Technique | MAE | ONNX | GitHub |
|---|-------|------|-----------|-----|------|--------|
| 1 | Gaze-LLE | 2024 | Transformer (DINOv2) | 3.7° | PyTorch | github.com/fkryan/gazelle |
| 2 | L2CS-Net | 2022 | CNN (ResNet, 2-branch) | 3.92° | Yes | github.com/ahmedkelqaq/L2CS-Net |
| 3 | GazeTR | 2022 | CNN+Transformer | 4.0° | Yes | github.com/yihuacheng/GazeTR |
| 4 | UniGaze | 2023 | ViT+MAE pretrain | 4.0° | Yes | github.com/unigaze/unigaze |
| 5 | PTGE | 2023 | ViT+calibration | 4.1° | Partial | - |
| 6 | InsightFace Gaze | 2023 | CNN (edge) | 4.5° | Yes | github.com/deepinsight/insightface |
| 7 | ETH-XGaze | 2020 | CNN (ResNet50) | 4.5° | Yes | github.com/xucong-zhang/ETH-XGaze |
| 8 | RT-GENE | 2018 | CNN ensemble | 4.8° | Yes | github.com/Tobias-Fischer/rt_gene |
| 9 | MPIIFaceGaze | 2017 | CNN (spatial) | 4.8° | Yes | github.com/ishtos/mpII-face-gaze |
| 10 | GazeMAE | 2020 | Autoencoder | 4.9° | Yes | github.com/chipbautista/gazemae |

## 실시간 CPU 추론 가능 (< 50ms)
- L2CS-Net (ResNet18/34), InsightFace Gaze, MPIIFaceGaze
- Transformer 기반 (Gaze-LLE, GazeTR)은 GPU 없으면 50ms 초과 가능

## 현재 사용 중
- L2CS-Net ResNet-34 (yakhyo/facial-analysis ONNX)
- SCRFD 2.5G face detection
