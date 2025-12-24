"""
Detection Service - Core logic for running YOLO and CNN inferences
"""
import numpy as np
import cv2
from PIL import Image
from utils.model_manager import load_yolo_model, load_cnn_model
from utils.traffic_sign_mapping import get_cnn_class_name, translate_sign_name
from config.settings import YOLO_MODELS_DIR, CNN_MODELS_DIR, CNN_INPUT_SIZE

def yolo_predict(image: Image.Image, model_name: str, models_dir: str):
    """Run YOLO detection"""
    model = load_yolo_model(model_name, models_dir)
    img_np = np.array(image)
    results = model(img_np)
    predictions = []
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name_en = model.names[class_id]
            class_name_vi = translate_sign_name(class_name_en)
            
            predictions.append({
                "box_coordinates": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name_vi,
                "cnn_confidence": 0.0 
            })
            
    return predictions

def cnn_predict(image: Image.Image, model_name: str, models_dir: str):
    """Run CNN Classification (MobileNetV2)"""
    model = load_cnn_model(model_name, models_dir)
    
    # Preprocess image for CNN
    img_resized = image.resize(CNN_INPUT_SIZE)
    img_array = np.array(img_resized)
    
    # --- QUAN TRỌNG: CHIA 255 ĐỂ KHỚP VỚI TRAINING ---
    img_array = img_array.astype('float32') / 255.0
    # -------------------------------------------------
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # Inference
    predictions = model.predict(img_array)
    predicted_class_id = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    
    # Map sang tên tiếng Việt
    class_name = get_cnn_class_name(predicted_class_id)
    
    return {
        "class_name": class_name,
        "class_id": predicted_class_id,
        "confidence": confidence,
        "box_coordinates": None
    }

def two_stage_predict(image: Image.Image, yolo_model_name: str, cnn_model_name: str, 
                     yolo_dir: str, cnn_dir: str, conf_threshold: float = 0.25):
    """Pipeline: YOLO Detect -> Crop -> CNN Classify"""
    yolo_results = yolo_predict(image, yolo_model_name, yolo_dir)
    final_predictions = []
    
    img_np = np.array(image)
    cnn_model = load_cnn_model(cnn_model_name, cnn_dir)
    
    for pred in yolo_results:
        if pred['confidence'] < conf_threshold:
            continue
            
        x1, y1, x2, y2 = pred['box_coordinates']
        
        # Crop có padding
        h, w, _ = img_np.shape
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1_c = max(0, x1 - pad_x)
        y1_c = max(0, y1 - pad_y)
        x2_c = min(w, x2 + pad_x)
        y2_c = min(h, y2 + pad_y)
        
        cropped_img = img_np[y1_c:y2_c, x1_c:x2_c]
        
        if cropped_img.size == 0:
            continue
            
        # Preprocess cho CNN (cũng phải chia 255)
        cropped_pil = Image.fromarray(cropped_img)
        img_resized = cropped_pil.resize(CNN_INPUT_SIZE)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Classify
        cnn_preds = cnn_model.predict(img_array)
        cnn_class_id = int(np.argmax(cnn_preds[0]))
        cnn_conf = float(np.max(cnn_preds[0]))
        class_name = get_cnn_class_name(cnn_class_id)
        
        pred['class_id'] = cnn_class_id
        pred['class_name'] = class_name
        pred['cnn_confidence'] = cnn_conf
        
        final_predictions.append(pred)
        
    return final_predictions