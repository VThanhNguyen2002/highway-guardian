# Highway Guardian 🛡️

**AI-Powered Traffic Infrastructure Management & Analytics Dashboard**

A production-grade system for automated Vietnamese traffic sign detection using a two-stage ensemble AI pipeline, with a real-time Firebase-backed analytics dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit App (Live Inference UI)                      │
│  • Real-time camera/video input                          │
│  • YOLOv8 tiling → MobileNetV2 ensemble                 │
└───────────────────┬─────────────────────────────────────┘
                    │ HTTP (FastAPI)
┌───────────────────▼─────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                           │
│  • POST /api/v1/detect  — async image upload + polling  │
│  • Two-stage inference (YOLOv8 + MobileNetV2)           │
│  • Firebase Firestore sync (sync_detection)              │
└───────────────────┬─────────────────────────────────────┘
                    │ Firestore onSnapshot
┌───────────────────▼─────────────────────────────────────┐
│  Vue 3 Dashboard  (port 5173)                           │
│  • Real-time KPIs, charts, detection logs               │
│  • CRUD: edit labels, delete false positives            │
│  • Profile page & role-based auth guard                  │
└─────────────────────────────────────────────────────────┘
```

## Two-Stage Inference Pipeline

| Stage | Model | Task |
|-------|-------|------|
| 1 — Detection | YOLOv8s (tiled 640×640, 20% overlap) | Locate sign bounding boxes |
| WBF | Weighted Boxes Fusion | Merge overlapping tile detections |
| 2 — Classification | MobileNetV2 (224×224 crops) | Classify into 7 Zalo AI 2020 classes |

## Tech Stack

| Layer | Technology |
|-------|-----------| 
| AI Models | YOLOv8 (Ultralytics), MobileNetV2 (PyTorch) |
| Backend | FastAPI, Uvicorn, ensemble-boxes |
| Inference UI | Streamlit, OpenCV |
| Dashboard | Vue 3, Vite, Chart.js, Vue Router, Pinia |
| Database | Firebase Firestore (real-time) |
| Auth | Firebase Authentication |

## Quickstart

```bash
# 1. Install Python dependencies (from project root)
pip install -r backend/requirements.txt

# 2. Place model weights (see models/README.md)
#    models/yolov8_v2.pt
#    models/best_mobilenet_v2.pth

# 3. Copy and fill environment file
cp .env.example .env

# 4a. One-command start (Linux / macOS)
bash run.sh

# 4b. One-command start (Windows)
run.bat

# --- OR manually in 3 terminals: ---
# Terminal 1 — Backend API
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Streamlit inference UI
cd streamlit_app && streamlit run app.py

# Terminal 3 — Vue Dashboard
cd frontend && npm install && npm run dev
```

## Project Structure

```
highway-guardian/
├── backend/          # FastAPI app, inference pipeline, API routes
├── frontend/         # Vue 3 dashboard (see frontend/README.md)
├── streamlit_app/    # Live inference UI
├── models/           # Model weights (git-ignored — see models/README.md)
├── src/              # Shared Python utilities (mapping, firebase_sync)
├── configs/          # Zalo class config, training YAML files
├── notebooks/        # Jupyter training notebooks (YOLO + CNN)
├── scripts/          # Dataset management & deployment utilities
├── keys/             # Firebase service account key (git-ignored)
├── uploads/          # Temporary image uploads (git-ignored)
├── data/             # Datasets (git-ignored)
├── run.sh            # Linux/macOS one-click startup
└── run.bat           # Windows one-click startup
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
# Firebase Admin SDK
FIREBASE_KEY_PATH=keys/firebase_key.json

# App config
DEBUG=true
CORS_ORIGINS=http://localhost:5173
```

The **frontend** uses separate `VITE_FIREBASE_*` variables — see `frontend/README.md`.
