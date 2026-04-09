"""
backend/core/inference_pipeline.py

Executes independent YOLOv8 or MobileNetV2 pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from backend.config.settings import Settings
from backend.core.model_loader import get_cnn, get_device, get_yolo
from backend.core.pdf_parser import PDFRuleParser
from src.core.mapping import NEW_2026_SIGN_IDS, SIGN_NAMES, get_sign_name

_NORMALISE_MEAN: list[float] = [0.485, 0.456, 0.406]
_NORMALISE_STD: list[float] = [0.229, 0.224, 0.225]


@dataclass
class DetectionResult:
    box_coordinates: Optional[list[int]]
    confidence: float
    class_id: int
    class_name: str
    is_valid: bool


class TrafficSignDetector:
    def __init__(self, settings: Settings, pdf_parser: PDFRuleParser) -> None:
        self._settings = settings
        self._pdf_parser = pdf_parser
        self._input_size: tuple[int, int] = settings.cnn_input_size

    def detect_yolo(
        self,
        image: Image.Image,
        confidence_threshold: Optional[float] = None,
    ) -> list[DetectionResult]:
        threshold: float = (
            confidence_threshold
            if confidence_threshold is not None
            else self._settings.default_confidence_threshold
        )

        if image.mode != "RGB":
            image = image.convert("RGB")

        img_rgb: np.ndarray = np.array(image)
        img_bgr: np.ndarray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        results: list[DetectionResult] = []

        yolo = get_yolo()
        yolo_output = yolo(img_bgr, verbose=False)

        for result in yolo_output:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                yolo_conf: float = float(box.conf[0])
                if yolo_conf < threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                class_name = get_sign_name(class_id)
                is_valid = self._validate(class_id)

                results.append(
                    DetectionResult(
                        box_coordinates=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=yolo_conf,
                        class_id=class_id,
                        class_name=class_name,
                        is_valid=is_valid,
                    )
                )

        del yolo_output
        return results

    def classify_cnn(self, image: Image.Image) -> DetectionResult:
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Letterbox / Pad to Square logic
        width, height = image.size
        max_dim = max(width, height)
        
        square_pad = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        
        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        square_pad.paste(image, (paste_x, paste_y))
            
        pil_crop = square_pad.resize(
            (self._input_size[1], self._input_size[0]),
            resample=Image.BILINEAR,
        )

        tensor = TF.to_tensor(pil_crop)
        tensor = TF.normalize(tensor, _NORMALISE_MEAN, _NORMALISE_STD)
        tensor = tensor.unsqueeze(0).to(get_device())

        cnn = get_cnn()
        with torch.no_grad():
            logits = cnn(tensor)
            temperature = 1.5
            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=1)
            confidence_tensor, pred_tensor = probabilities.max(dim=1)

        folder_names = [str(i) for i in range(103)]
        folder_names.sort() 
        
        pred_idx = int(pred_tensor.item())
        real_class_id = int(folder_names[pred_idx])
        confidence: float = float(confidence_tensor.item())
        
        class_name = get_sign_name(real_class_id)
        is_valid = self._validate(real_class_id)
        
        return DetectionResult(
            box_coordinates=[],
            confidence=confidence,
            class_id=real_class_id,
            class_name=class_name,
            is_valid=is_valid,
        )

    def _validate(self, class_id: int) -> bool:
        if class_id in NEW_2026_SIGN_IDS:
            return True

        if class_id in SIGN_NAMES:
            category_prefixes: dict[int, str] = {
                1: "P.102",
                2: "P.130",
                3: "P.123a",
                4: "P.127",
                5: "P.125",
                6: "W.201a",
                7: "R.301c",
            }
            representative_code = category_prefixes.get(class_id, "")
            if representative_code:
                return self._pdf_parser.is_sign_valid(representative_code)

        return False