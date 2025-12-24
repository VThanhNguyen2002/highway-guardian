# 🔧 Cập nhật CNN Class ID Mapping

## ⚠️ Quan trọng

File `src/utils/vietnam_traffic_signs.py` đã được tạo với YOLO mapping hoàn chỉnh, nhưng **CNN mapping cần được cập nhật** dựa trên thứ tự training của bạn.

## 📝 Cách cập nhật

### Bước 1: Xác định thứ tự classes trong CNN training

Kiểm tra file training code hoặc model metadata để xem thứ tự classes:

```python
# Ví dụ từ training code
FINAL_MASTER_CLASSES = [
    'DP.135', 'P.102', 'P.103a', 'P.103b', 'P.103c', 'P.104', 'P.106a', 'P.106b',
    'P.107a', 'P.111', 'P.112', 'P.115', 'P.117', 'P.123a', 'P.123b', 'P.124a',
    # ... etc
]
```

### Bước 2: Tạo mapping dictionary

Mở file `src/utils/vietnam_traffic_signs.py` và tìm dòng:

```python
CNN_CLASS_ID_TO_CODE = {
    # This will be populated based on your CNN training order
    # Example structure (you need to provide the actual mapping):
    # 0: 'P.102',
    # 1: 'P.103a',
    # ... etc
}
```

### Bước 3: Điền mapping

Thay thế bằng mapping thực tế:

```python
CNN_CLASS_ID_TO_CODE = {
    0: 'DP.135',
    1: 'P.102',
    2: 'P.103a',
    3: 'P.103b',
    4: 'P.103c',
    5: 'P.104',
    6: 'P.106a',
    7: 'P.106b',
    8: 'P.107a',
    9: 'P.111',
    10: 'P.112',
    11: 'P.115',
    12: 'P.117',
    13: 'P.123a',
    14: 'P.123b',
    15: 'P.124a',
    16: 'P.124b',
    17: 'P.124c',
    18: 'P.124d',
    19: 'P.125',
    20: 'P.126',
    21: 'P.127',
    22: 'P.128',
    23: 'P.129',
    24: 'P.130',
    25: 'P.131a',
    26: 'P.137',
    27: 'P.139',
    28: 'P.245a',
    29: 'R.122',
    30: 'R.301a',
    31: 'R.301c',
    32: 'R.301d',
    33: 'R.301e',
    34: 'R.302a',
    35: 'R.302b',
    36: 'R.303',
    37: 'R.306',
    38: 'R.407a',
    39: 'R.409',
    40: 'R.415a',
    41: 'R.425',
    42: 'R.434',
    43: 'S.505a',
    44: 'S.509a',
    45: 'W.201a',
    46: 'W.201b',
    47: 'W.202a',
    48: 'W.202b',
    49: 'W.203a',
    50: 'W.203b',
    51: 'W.203c',
    52: 'W.205a',
    53: 'W.205b',
    54: 'W.205c',
    55: 'W.205d',
    56: 'W.207',
    57: 'W.207a',
    58: 'W.207b',
    59: 'W.207c',
    60: 'W.208',
    61: 'W.209',
    62: 'W.210',
    63: 'W.219',
    64: 'W.221b',
    65: 'W.222a',
    66: 'W.224',
    67: 'W.225',
    68: 'W.227',
    69: 'W.233',
    70: 'W.235',
    71: 'W.239b',
    72: 'W.245a',
    73: 'W.246a',
    74: 'W.246c',
}
```

### Bước 4: Test

Sau khi cập nhật, test bằng cách:

```bash
# Start backend
cd src
python main.py

# Test prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "model_name=bien_bao_mobilenetv2_MERGED_BALANCED_model (1).h5" \
  -F "model_type=cnn"
```

Kết quả sẽ hiển thị:
```json
{
  "predictions": [{
    "class_name": "P.102: Cấm đi ngược chiều",
    "sign_code": "P.102",
    "category": "Biển cấm",
    "confidence": 0.95,
    "class_id": 1
  }]
}
```

## 🎯 Kết quả mong đợi

Sau khi cập nhật, hệ thống sẽ hiển thị:
- **Frontend**: "P.102: Cấm đi ngược chiều" thay vì "Class_1"
- **API Response**: Bao gồm sign_code, category, và full display name
- **Bounding boxes**: Label hiển thị mã và tên biển báo

## 📚 Tham khảo

- File mapping: `src/utils/vietnam_traffic_signs.py`
- Detection service: `src/services/detection_service.py`
- QCVN 41:2019: `docs/quy-chuan-ky-thuat-qcvn-41-2019-bgtvt-bao-hieu-duong-bo.pdf`

---

*Sau khi cập nhật, hệ thống sẽ tự động sử dụng mapping mới!*
