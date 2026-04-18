# Models Directory

This directory holds the AI model weight files used by the Two-Stage Inference Pipeline.

> ⚠️ **Model weights are NOT tracked by Git** (excluded via `.gitignore`).  
> You must download and place them here manually before starting the backend.

---

## Required Files

| Filename | Model | Stage | Size (approx.) |
|---|---|---|---|
| `yolov8_v2.pt` | YOLOv8s (Ultralytics) | Stage 1 — Detection (tiled 640×640) | ~22 MB |
| `best_mobilenet_v2.pth` | MobileNetV2 (PyTorch) | Stage 2 — Classification (8 classes) | ~14 MB |

## Pipeline Role

```
Image Input
   → Tiling (640×640, 20% overlap)
       → yolov8_v2.pt          ← Stage 1: locate bounding boxes
   → Weighted Boxes Fusion (WBF)
       → best_mobilenet_v2.pth ← Stage 2: classify each cropped box
   → Firestore sync
```

## MobileNetV2 Class Map

The MobileNetV2 model outputs 8 logits (index 0 = Background, 1–7 = Zalo AI 2020 classes):

| ID | Label |
|----|-------|
| 0  | Background (filtered out) |
| 1  | Cấm ngược chiều |
| 2  | Cấm dừng và đỗ |
| 3  | Cấm rẽ |
| 4  | Giới hạn tốc độ |
| 5  | Cấm ô tô |
| 6  | Cấm đỗ |
| 7  | Cấm các phương tiện khác |

Source: `src/core/mapping.py` → `SIGN_NAMES`

## Obtaining the Weights

These weights were trained on the **Zalo AI 2020 Traffic Sign Dataset**.  
Contact the project maintainer or train from scratch using the notebooks in `notebooks/`:
- `notebooks/train-yolo.ipynb` — YOLOv8 training
- `notebooks/train-cnn.ipynb` — MobileNetV2 training
