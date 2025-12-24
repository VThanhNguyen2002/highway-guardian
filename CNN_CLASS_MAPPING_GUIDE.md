# Hướng dẫn Cập nhật CNN Class Mapping

## 🎯 Mục đích

File này hướng dẫn cách cập nhật mapping giữa CNN class IDs và tên biển báo tiếng Việt.

## 📍 File cần chỉnh sửa

**`src/utils/traffic_sign_mapping.py`**

## 🔍 Cách xác định Class IDs

### Bước 1: Kiểm tra Training Code

Tìm file training code của CNN model (thường là Jupyter notebook hoặc Python script):

```python
# Ví dụ trong training code
class_names = [
    'Cấm rẽ trái',
    'Cấm rẽ phải', 
    'Giới hạn tốc độ 50',
    # ...
]
```

### Bước 2: Kiểm tra Model Metadata

Nếu model có metadata:

```python
import tensorflow as tf

model = tf.keras.models.load_model('path/to/model.h5')
# Kiểm tra model.layers, model.get_config(), etc.
```

### Bước 3: Test với Sample Images

Tạo script test:

```python
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('models/cnn/your_model.h5')

# Test với ảnh mẫu
test_images = {
    'cam_re_trai.jpg': 'Cấm rẽ trái',
    'cam_re_phai.jpg': 'Cấm rẽ phải',
    # ... thêm ảnh test
}

for img_path, expected_name in test_images.items():
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_id = np.argmax(predictions[0])
    
    print(f"{img_path} -> Class ID: {class_id} (Expected: {expected_name})")
```

## ✏️ Cập nhật Mapping

### File: `src/utils/traffic_sign_mapping.py`

```python
CNN_CLASS_NAMES = {
    # === Biển CẤM ===
    0: "Cấm rẽ trái",
    1: "Cấm rẽ phải",
    2: "Cấm quay đầu",
    3: "Cấm đỗ xe",
    4: "Cấm dừng xe",
    5: "Cấm đi ngược chiều",
    
    # === Biển GIỚI HẠN TỐC ĐỘ ===
    10: "Giới hạn tốc độ 20",
    11: "Giới hạn tốc độ 30",
    12: "Giới hạn tốc độ 40",
    13: "Giới hạn tốc độ 50",
    14: "Giới hạn tốc độ 60",
    15: "Giới hạn tốc độ 70",
    16: "Giới hạn tốc độ 80",
    17: "Giới hạn tốc độ 100",
    
    # === Biển HIỆU LỆNH ===
    20: "Đi thẳng",
    21: "Rẽ phải",
    22: "Rẽ trái",
    23: "Đi thẳng hoặc rẽ phải",
    24: "Đi thẳng hoặc rẽ trái",
    25: "Đi bên phải",
    26: "Đi bên trái",
    27: "Vòng xuyến",
    
    # === Biển CẢNH BÁO ===
    30: "Nguy hiểm khác",
    31: "Chỗ ngoặt nguy hiểm",
    32: "Giao nhau",
    33: "Đường người đi bộ cắt ngang",
    34: "Trẻ em",
    35: "Công trường",
    36: "Đường trơn trượt",
    37: "Gờ giảm tốc",
    
    # === Biển CHỈ DẪN ===
    40: "Bãi đỗ xe",
    41: "Trạm xăng",
    42: "Bệnh viện",
    43: "Nhà hàng",
    44: "Khách sạn",
    
    # Thêm các class khác...
}
```

## 🎨 Phân loại theo Nhóm

Bạn cũng có thể thêm function để phân loại:

```python
def get_cnn_sign_category(class_id: int) -> str:
    """Get sign category from CNN class ID"""
    if 0 <= class_id < 10:
        return "Biển cấm"
    elif 10 <= class_id < 20:
        return "Giới hạn tốc độ"
    elif 20 <= class_id < 30:
        return "Biển hiệu lệnh"
    elif 30 <= class_id < 40:
        return "Biển cảnh báo"
    elif 40 <= class_id < 50:
        return "Biển chỉ dẫn"
    else:
        return "Khác"
```

## 🧪 Test Mapping

Tạo file test: `src/tests/test_mapping.py`

```python
from utils.traffic_sign_mapping import get_cnn_class_name

# Test cases
test_cases = [
    (0, "Cấm rẽ trái"),
    (13, "Giới hạn tốc độ 50"),
    (20, "Đi thẳng"),
    # ... thêm test cases
]

for class_id, expected_name in test_cases:
    actual_name = get_cnn_class_name(class_id)
    status = "✓" if actual_name == expected_name else "✗"
    print(f"{status} Class {class_id}: {actual_name} (Expected: {expected_name})")
```

## 📊 Template Excel/CSV

Tạo file `cnn_class_mapping.csv`:

```csv
class_id,class_name_vi,class_name_en,category
0,Cấm rẽ trái,No left turn,prohibitory
1,Cấm rẽ phải,No right turn,prohibitory
2,Cấm quay đầu,No U-turn,prohibitory
...
```

Sau đó convert sang Python dict:

```python
import pandas as pd

df = pd.read_csv('cnn_class_mapping.csv')
mapping = dict(zip(df['class_id'], df['class_name_vi']))
print(mapping)
```

## 🔄 Workflow Cập nhật

1. **Xác định classes** từ training code
2. **Test với sample images** để verify
3. **Cập nhật** `CNN_CLASS_NAMES` trong `traffic_sign_mapping.py`
4. **Test API** với `/predict` endpoint
5. **Verify** trên frontend Camera/Detect pages

## 📝 Lưu ý

- Class IDs phải match với model training
- Tên tiếng Việt nên ngắn gọn, dễ đọc
- Có thể thêm cả tên tiếng Anh nếu cần
- Nên group theo categories để dễ quản lý

## 🎯 Ví dụ Hoàn chỉnh

Dựa trên GTSRB dataset (43 classes):

```python
CNN_CLASS_NAMES = {
    0: "Giới hạn tốc độ 20km/h",
    1: "Giới hạn tốc độ 30km/h",
    2: "Giới hạn tốc độ 50km/h",
    3: "Giới hạn tốc độ 60km/h",
    4: "Giới hạn tốc độ 70km/h",
    5: "Giới hạn tốc độ 80km/h",
    6: "Hết giới hạn tốc độ 80km/h",
    7: "Giới hạn tốc độ 100km/h",
    8: "Giới hạn tốc độ 120km/h",
    9: "Cấm vượt",
    10: "Cấm xe tải vượt",
    11: "Giao nhau có đường ưu tiên",
    12: "Đường ưu tiên",
    13: "Nhường đường",
    14: "Dừng lại",
    15: "Cấm xe cơ giới",
    16: "Cấm xe tải",
    17: "Cấm đi vào",
    18: "Nguy hiểm chung",
    19: "Chỗ ngoặt nguy hiểm bên trái",
    20: "Chỗ ngoặt nguy hiểm bên phải",
    21: "Chỗ ngoặt kép",
    22: "Đường gồ ghề",
    23: "Đường trơn",
    24: "Đường hẹp bên phải",
    25: "Công trường",
    26: "Đèn tín hiệu",
    27: "Người đi bộ",
    28: "Trẻ em qua đường",
    29: "Xe đạp qua đường",
    30: "Cẩn thận băng tuyết",
    31: "Động vật hoang dã",
    32: "Hết tất cả giới hạn",
    33: "Rẽ phải phía trước",
    34: "Rẽ trái phía trước",
    35: "Chỉ đi thẳng",
    36: "Đi thẳng hoặc rẽ phải",
    37: "Đi thẳng hoặc rẽ trái",
    38: "Đi bên phải",
    39: "Đi bên trái",
    40: "Vòng xuyến bắt buộc",
    41: "Hết cấm vượt",
    42: "Hết cấm xe tải vượt",
}
```

---

*Cập nhật: 2025-01-21*
