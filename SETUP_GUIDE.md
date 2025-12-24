# 🚀 Setup Guide - Highway Guardian

## 📋 Prerequisites

### 1. Python 3.8+
**Download**: https://www.python.org/downloads/

**Cài đặt**:
- ✅ Check "Add Python to PATH"
- ✅ Install pip
- ✅ Install for all users (optional)

**Verify**:
```bash
python --version
# hoặc
py --version
```

### 2. Node.js 18+
**Download**: https://nodejs.org/

**Verify**:
```bash
node --version
npm --version
```

### 3. Git (Optional)
**Download**: https://git-scm.com/

---

## 🔧 Backend Setup

### Bước 1: Cài đặt Dependencies

```bash
cd src
pip install -r requirements.txt
```

**Nếu gặp lỗi**, thử:
```bash
# Windows
py -m pip install -r requirements.txt

# Hoặc với Python 3 cụ thể
python3 -m pip install -r requirements.txt
```

### Bước 2: Kiểm tra Models

Đảm bảo có models trong thư mục:
```
models/
├── yolo/
│   └── best.pt
└── cnn/
    └── bien_bao_mobilenetv2_MERGED_BALANCED_model (1).h5
```

### Bước 3: Start Backend

**Option 1: Batch file (Windows)**
```bash
start_backend.bat
```

**Option 2: Manual**
```bash
cd src
python main.py
```

**Option 3: Uvicorn**
```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Bước 4: Verify

Mở browser: http://localhost:8000

Hoặc test với curl:
```bash
curl http://localhost:8000/
curl http://localhost:8000/models
```

---

## 🎨 Frontend Setup

### Bước 1: Cài đặt Dependencies

```bash
cd frontend
npm install
```

### Bước 2: Start Dev Server

```bash
npm run dev
```

### Bước 3: Verify

Mở browser: http://localhost:5173

---

## 🐛 Troubleshooting

### Backend không start

#### Lỗi: "Python not found"
**Giải pháp**:
1. Cài đặt Python từ python.org
2. Thêm Python vào PATH
3. Restart terminal

#### Lỗi: "Module not found"
**Giải pháp**:
```bash
pip install -r src/requirements.txt
```

#### Lỗi: "Port 8000 already in use"
**Giải pháp**:
```bash
# Tìm process đang dùng port 8000
netstat -ano | findstr :8000

# Kill process
taskkill /PID <PID> /F
```

#### Lỗi: "Model not found"
**Giải pháp**:
- Kiểm tra file models có tồn tại
- Kiểm tra đường dẫn trong `src/config/settings.py`

### Frontend không start

#### Lỗi: "npm not found"
**Giải pháp**:
1. Cài đặt Node.js từ nodejs.org
2. Restart terminal

#### Lỗi: "Port 5173 already in use"
**Giải pháp**:
```bash
# Kill process
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

#### Lỗi: "Connection refused to localhost:8000"
**Giải pháp**:
- Đảm bảo backend đang chạy
- Kiểm tra CORS settings trong `src/config/settings.py`

### Firebase Authentication

#### Lỗi: "Firebase config not found"
**Giải pháp**:
- Kiểm tra file `frontend/.env`
- Đảm bảo có đầy đủ Firebase credentials

---

## 📦 Production Deployment

### Docker (Recommended)

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Stop
docker-compose down
```

### Manual Deployment

#### Backend:
```bash
cd src
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Frontend:
```bash
cd frontend
npm run build
# Deploy dist/ folder to web server
```

---

## 🔐 Environment Variables

### Backend (`src/config/settings.py`)
```python
API_HOST = "0.0.0.0"
API_PORT = 8000
YOLO_MODELS_DIR = "/path/to/models/yolo"
CNN_MODELS_DIR = "/path/to/models/cnn"
```

### Frontend (`frontend/.env`)
```env
VITE_FIREBASE_API_KEY="your-api-key"
VITE_FIREBASE_AUTH_DOMAIN="your-domain"
VITE_FIREBASE_PROJECT_ID="your-project-id"
# ... other Firebase configs
```

---

## ✅ Verification Checklist

### Backend:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models exist in `models/` folder
- [ ] Server starts without errors
- [ ] http://localhost:8000/ returns status
- [ ] http://localhost:8000/models returns model list

### Frontend:
- [ ] Node.js 18+ installed
- [ ] Dependencies installed (`npm install`)
- [ ] Dev server starts without errors
- [ ] http://localhost:5173/ loads
- [ ] Can login with Firebase
- [ ] Can access Detect and Camera pages

### Integration:
- [ ] Frontend can fetch models from backend
- [ ] Image upload works on Detect page
- [ ] Camera detection works on Camera page
- [ ] Toast notifications appear on login

---

## 📚 Next Steps

1. ✅ Setup complete
2. 📖 Read [TWO_STAGE_DETECTION_GUIDE.md](frontend/TWO_STAGE_DETECTION_GUIDE.md)
3. 🎨 Update CNN class mappings (see [CNN_CLASS_MAPPING_GUIDE.md](CNN_CLASS_MAPPING_GUIDE.md))
4. 🧪 Test with sample images
5. 🚀 Deploy to production

---

## 💡 Tips

### Development:
- Use `--reload` flag for auto-restart on code changes
- Check browser console for frontend errors
- Check terminal for backend errors

### Performance:
- Models are cached after first load
- Use YOLO Only mode for faster detection
- Reduce camera resolution for better FPS

### Debugging:
- Enable verbose logging in backend
- Use browser DevTools Network tab
- Check API responses in Network tab

---

*Happy Coding! 🎉*
