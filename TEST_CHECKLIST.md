# ✅ Test Checklist - Highway Guardian

## 🔍 Pre-deployment Checks

### 1. Backend Tests

#### ✅ API Endpoints
- [ ] `GET /` - Health check
- [ ] `GET /models` - List models
- [ ] `POST /predict` - YOLO prediction
- [ ] `POST /predict` - CNN prediction
- [ ] `POST /predict_two_stage` - Two-stage pipeline

#### ✅ Model Loading
- [ ] YOLO model loads successfully
- [ ] CNN model loads successfully
- [ ] Model caching works
- [ ] LRU eviction works

#### ✅ Error Handling
- [ ] Invalid model name returns 404
- [ ] Invalid model type returns 400
- [ ] Missing file returns error
- [ ] Corrupted image returns error

### 2. Frontend Tests

#### ✅ Authentication
- [ ] Login page loads
- [ ] Firebase auth works
- [ ] Toast notification appears on success
- [ ] Toast notification appears on error
- [ ] Redirect to Detect after login
- [ ] Logout works

#### ✅ Detect Page
- [ ] Model type selector works
- [ ] Model list loads from API
- [ ] File upload works
- [ ] Image preview displays
- [ ] Prediction works (YOLO)
- [ ] Prediction works (CNN)
- [ ] Bounding boxes draw correctly
- [ ] Results list displays
- [ ] Loading states work

#### ✅ Camera Page
- [ ] Camera permission request
- [ ] Video stream displays
- [ ] YOLO Only mode works
- [ ] YOLO + CNN mode works
- [ ] FPS counter updates
- [ ] Detection count updates
- [ ] Bounding boxes draw correctly
- [ ] Stop camera works

### 3. Integration Tests

#### ✅ Backend ↔ Frontend
- [ ] CORS allows frontend requests
- [ ] API responses match expected format
- [ ] Error messages display correctly
- [ ] Loading states sync with API calls

#### ✅ Model Pipeline
- [ ] YOLO detection returns boxes
- [ ] CNN classification returns class
- [ ] Two-stage pipeline works end-to-end
- [ ] Confidence thresholds work

### 4. Performance Tests

#### ✅ Speed
- [ ] First model load < 3s
- [ ] Cached model load < 0.2s
- [ ] YOLO prediction < 1s
- [ ] CNN prediction < 1s
- [ ] Two-stage prediction < 2s

#### ✅ Memory
- [ ] Model cache doesn't exceed limit
- [ ] LRU eviction frees memory
- [ ] No memory leaks

#### ✅ Real-time
- [ ] Camera FPS > 10 (YOLO Only)
- [ ] Camera FPS > 5 (Two-stage)
- [ ] No frame drops

### 5. UI/UX Tests

#### ✅ Visual
- [ ] Gradient colors display correctly
- [ ] Animations smooth
- [ ] Responsive on different screens
- [ ] Icons display correctly
- [ ] Fonts load correctly

#### ✅ Interaction
- [ ] Buttons clickable
- [ ] Dropdowns work
- [ ] File upload drag & drop
- [ ] Toast auto-closes
- [ ] Loading spinners show

---

## 🔐 Security Checks

### Backend
- [ ] CORS configured correctly
- [ ] No sensitive data in responses
- [ ] File upload size limited
- [ ] Input validation works

### Frontend
- [ ] Firebase credentials in .env
- [ ] No API keys in code
- [ ] Auth tokens secure
- [ ] XSS protection

---

## 📊 Test Results Template

### Test Run: [Date]

| Category | Passed | Failed | Notes |
|----------|--------|--------|-------|
| Backend API | 0/5 | 0/5 | Need Python |
| Frontend | 0/10 | 0/10 | - |
| Integration | 0/4 | 0/4 | - |
| Performance | 0/8 | 0/8 | - |
| UI/UX | 0/10 | 0/10 | - |

**Total**: 0/37 tests

---

## 🐛 Known Issues

1. **Python not installed**
   - Status: ⚠️ Blocking
   - Solution: Install Python 3.8+
   - Priority: High

2. **Backend not running**
   - Status: ⚠️ Blocking
   - Solution: Start backend after Python install
   - Priority: High

---

## ✅ Manual Test Script

### Backend Test (After Python install):

```bash
# 1. Start backend
cd src
python main.py

# 2. Test health check
curl http://localhost:8000/

# 3. Test models endpoint
curl http://localhost:8000/models

# 4. Test prediction (need image file)
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "model_name=best.pt" \
  -F "model_type=yolo"
```

### Frontend Test:

```bash
# 1. Open browser
http://localhost:5173

# 2. Test login
- Enter email/password
- Check toast notification
- Verify redirect

# 3. Test Detect page
- Select YOLO model
- Upload image
- Click "Bắt đầu Nhận diện"
- Verify results

# 4. Test Camera page
- Select mode
- Select models
- Click "Bật Camera"
- Allow camera permission
- Verify real-time detection
```

---

## 📝 Test Automation (Future)

### Unit Tests
```python
# tests/test_model_manager.py
def test_load_yolo_model():
    model = load_yolo_model("best.pt", "models/yolo")
    assert model is not None

def test_model_caching():
    model1 = load_yolo_model("best.pt", "models/yolo")
    model2 = load_yolo_model("best.pt", "models/yolo")
    assert model1 is model2  # Same instance
```

### Integration Tests
```python
# tests/test_api.py
def test_predict_endpoint():
    response = client.post("/predict", ...)
    assert response.status_code == 200
    assert "predictions" in response.json()
```

---

*Last Updated: 2025-01-21*
