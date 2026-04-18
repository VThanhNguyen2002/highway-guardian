"""
backend/core/inference_pipeline.py

Two-Stage Pipeline (v2)
========================
Original Image
    → Tiling (640×640, 20% overlap)
        → YOLOv8 inference on each tile
    → Map tile boxes back to original-image coordinates (normalised 0–1)
    → Weighted Boxes Fusion (WBF) — merge overlapping YOLO boxes
    → Crop each merged box from the original image
    → MobileNetV2 classifies each crop → 7 Zalo AI 2020 classes
    → sync_detection() pushes result to Firestore
    → Emit final list[DetectionResult]

Legacy yolo-only and cnn-only modes are preserved for backwards
compatibility with existing API clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image

from backend.config.settings import Settings
from backend.core.model_loader import get_cnn, get_device, get_yolo
from backend.core.pdf_parser import PDFRuleParser
from src.core.mapping import VALID_CLASS_IDS, get_sign_name, is_valid_class
from src.utils.firebase_sync import sync_detection

_NORMALISE_MEAN: list[float] = [0.485, 0.456, 0.406]
_NORMALISE_STD:  list[float] = [0.229, 0.224, 0.225]

# Minimum bounding-box area (px²) after WBF — discard degenerate crops.
_MIN_BOX_AREA: int = 4


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

    # =========================================================================
    # Public inference entry points
    # =========================================================================

    def detect_yolo(
        self,
        image: Image.Image,
        confidence_threshold: Optional[float] = None,
        user_display_name: str = "admin",
    ) -> list[DetectionResult]:
        """YOLOv8-only detection (single pass, no tiling)."""
        threshold: float = (
            confidence_threshold
            if confidence_threshold is not None
            else self._settings.default_confidence_threshold
        )

        if image.mode != "RGB":
            image = image.convert("RGB")

        img_bgr: np.ndarray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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
                is_val = is_valid_class(class_id)

                detection = DetectionResult(
                    box_coordinates=[int(x1), int(y1), int(x2), int(y2)],
                    confidence=yolo_conf,
                    class_id=class_id,
                    class_name=class_name,
                    is_valid=is_val,
                )
                results.append(detection)

                # ── Firebase Sync ─────────────────────────────────────────────
                sync_detection(
                    data={
                        "label":           class_name,
                        "confidence":      yolo_conf,
                        "class_id":        class_id,
                        "box_coordinates": [int(x1), int(y1), int(x2), int(y2)],
                        "is_valid":        is_val,
                        "model_used":      "YOLOv8-Single",
                        "image_path":      "",
                    },
                    display_name=user_display_name,
                )

        del yolo_output
        return results

    def classify_cnn(
        self,
        image: Image.Image,
        user_display_name: str = "admin",
    ) -> DetectionResult:
        """MobileNetV2-only classification on the full image (letterboxed)."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        max_dim = max(width, height)
        square_pad = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_pad.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
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
            probs = torch.softmax(logits / 1.5, dim=1)
            confidence_t, pred_t = probs.max(dim=1)

        class_id = int(pred_t.item())      # 0–7
        confidence = float(confidence_t.item())
        class_name = get_sign_name(class_id)
        is_val = is_valid_class(class_id)

        # ── Firebase Sync ─────────────────────────────────────────────────────
        sync_detection(
            data={
                "label":           class_name,
                "confidence":      confidence,
                "class_id":        class_id,
                "box_coordinates": [],
                "is_valid":        is_val,
                "model_used":      "MobileNetV2-Single",
                "image_path":      "",
            },
            display_name=user_display_name,
        )

        return DetectionResult(
            box_coordinates=[],
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            is_valid=is_val,
        )

    def detect_ensemble(
        self,
        image: Image.Image,
        user_display_name: str = "admin",
    ) -> list[DetectionResult]:
        """
        Two-Stage Pipeline (v2).

        Stage 1 — YOLO tiling:
            • Cut the full image into 640×640 tiles (20% overlap).
            • Run YOLOv8 on every tile.
            • Re-project each tile's boxes to full-image pixel coords.
            • Normalise coordinates to [0, 1].
            • Merge all boxes with Weighted Boxes Fusion (WBF).

        Stage 2 — MobileNetV2 classification:
            • Crop each merged box from the original image.
            • Pad to square and resize to 224×224.
            • Classify with MobileNetV2 (7 Zalo classes, index 0 = background).
            • Push each successful detection to Firestore via sync_detection().

        Args:
            image: PIL image (any mode — converted to RGB internally).
            user_display_name: Username from the authenticated Firestore user doc.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_np: np.ndarray = np.array(image)       # H×W×3, RGB uint8
        img_h, img_w = img_np.shape[:2]

        # ── Stage 1: tile-based YOLO detection ───────────────────────────────
        tiles = self._tile_image(img_np)
        yolo_boxes, yolo_scores, yolo_labels = self._run_yolo_on_tiles(
            tiles, img_w, img_h
        )

        if not yolo_boxes:
            return []

        # ── WBF merge ────────────────────────────────────────────────────────
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            yolo_boxes,
            yolo_scores,
            yolo_labels,
            iou_thr=self._settings.wbf_iou_threshold,
            skip_box_thr=self._settings.wbf_conf_threshold,
        )

        # ── Stage 2: classify each crop with MobileNetV2 ─────────────────────
        results: list[DetectionResult] = []
        threshold = self._settings.default_confidence_threshold

        for box_norm, score, _label in zip(merged_boxes, merged_scores, merged_labels):
            if score < threshold:
                continue

            # Denormalise back to pixel coordinates.
            x1 = int(max(0, box_norm[0] * img_w))
            y1 = int(max(0, box_norm[1] * img_h))
            x2 = int(min(img_w, box_norm[2] * img_w))
            y2 = int(min(img_h, box_norm[3] * img_h))

            if (x2 - x1) * (y2 - y1) < _MIN_BOX_AREA:
                continue

            crop_pil = Image.fromarray(img_np[y1:y2, x1:x2])
            cnn_result = self._classify_crop(crop_pil)

            # Skip background predictions (class_id == 0)
            if cnn_result.class_id == 0:
                continue

            detection = DetectionResult(
                box_coordinates=[x1, y1, x2, y2],
                confidence=cnn_result.confidence,
                class_id=cnn_result.class_id,
                class_name=cnn_result.class_name,
                is_valid=cnn_result.is_valid,
            )
            results.append(detection)

            # ── Firebase Sync ─────────────────────────────────────────────────
            sync_detection(
                data={
                    "label":           cnn_result.class_name,
                    "confidence":      cnn_result.confidence,
                    "class_id":        cnn_result.class_id,
                    "box_coordinates": [x1, y1, x2, y2],
                    "is_valid":        cnn_result.is_valid,
                    "model_used":      "YOLOv8+MobileNetV2",
                    "image_path":      "",
                },
                display_name=user_display_name,
            )

        return results

    # =========================================================================
    # Private helpers — tiling
    # =========================================================================

    def _tile_image(
        self, img_np: np.ndarray
    ) -> list[tuple[np.ndarray, int, int]]:
        """
        Slice ``img_np`` into overlapping square tiles.

        Returns:
            List of ``(tile_array, offset_x, offset_y)`` where offsets are
            the top-left corner of the tile in original-image pixel coords.
        """
        tile_size: int = self._settings.tile_size
        overlap: float = self._settings.tile_overlap
        stride: int = max(1, int(tile_size * (1.0 - overlap)))

        img_h, img_w = img_np.shape[:2]
        tiles: list[tuple[np.ndarray, int, int]] = []

        y = 0
        while y < img_h:
            x = 0
            while x < img_w:
                x_end = min(x + tile_size, img_w)
                y_end = min(y + tile_size, img_h)
                tile = img_np[y:y_end, x:x_end]

                # Pad to tile_size × tile_size if needed (bottom-right padding).
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[: tile.shape[0], : tile.shape[1]] = tile
                    tile = padded

                tiles.append((tile, x, y))

                if x_end == img_w:
                    break
                x += stride

            if y_end == img_h:
                break
            y += stride

        return tiles

    # =========================================================================
    # Private helpers — YOLO tile inference
    # =========================================================================

    def _run_yolo_on_tiles(
        self,
        tiles: list[tuple[np.ndarray, int, int]],
        img_w: int,
        img_h: int,
    ) -> tuple[list[list[float]], list[list[float]], list[list[int]]]:
        """
        Run YOLOv8 on each tile and reproject boxes to original-image
        coordinates, then normalise to [0, 1].

        Returns WBF-compatible lists (one sub-list per model slot).
        """
        yolo = get_yolo()
        conf_thr = self._settings.wbf_conf_threshold

        agg_boxes:  list[list[float]] = []
        agg_scores: list[float]       = []
        agg_labels: list[int]         = []

        for tile_rgb, ox, oy in tiles:
            tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
            output = yolo(tile_bgr, verbose=False)

            for res in output:
                boxes_np = res.boxes.cpu().numpy()
                for box in boxes_np:
                    conf = float(box.conf[0])
                    if conf < conf_thr:
                        continue

                    tx1, ty1, tx2, ty2 = box.xyxy[0]

                    # Move tile coords → full-image pixel coords.
                    fx1 = float(ox) + float(tx1)
                    fy1 = float(oy) + float(ty1)
                    fx2 = float(ox) + float(tx2)
                    fy2 = float(oy) + float(ty2)

                    # Normalise strictly to [0, 1].
                    nx1 = max(0.0, min(1.0, fx1 / img_w))
                    ny1 = max(0.0, min(1.0, fy1 / img_h))
                    nx2 = max(0.0, min(1.0, fx2 / img_w))
                    ny2 = max(0.0, min(1.0, fy2 / img_h))

                    if nx2 <= nx1 or ny2 <= ny1:
                        continue

                    agg_boxes.append([nx1, ny1, nx2, ny2])
                    agg_scores.append(conf)
                    agg_labels.append(int(box.cls[0]))

            del output

        if not agg_boxes:
            return [], [], []

        # WBF expects one list-of-lists per model slot.
        return [agg_boxes], [agg_scores], [agg_labels]

    # =========================================================================
    # Private helpers — Stage-2 classification
    # =========================================================================

    def _classify_crop(self, crop: Image.Image) -> DetectionResult:
        """
        Pad a cropped bbox to square, resize to 224×224, and classify with
        MobileNetV2 (8 outputs: index 0 = background, 1–7 = Zalo classes).
        """
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        w, h = crop.size
        max_dim = max(w, h)
        padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        padded.paste(crop, ((max_dim - w) // 2, (max_dim - h) // 2))
        resized = padded.resize(
            (self._input_size[1], self._input_size[0]),
            resample=Image.BILINEAR,
        )

        tensor = TF.to_tensor(resized)
        tensor = TF.normalize(tensor, _NORMALISE_MEAN, _NORMALISE_STD)
        tensor = tensor.unsqueeze(0).to(get_device())

        cnn = get_cnn()
        with torch.no_grad():
            logits = cnn(tensor)
            probs = torch.softmax(logits / 1.5, dim=1)         # temperature scaling
            confidence_t, pred_t = probs.max(dim=1)

        class_id  = int(pred_t.item())       # 0–7
        confidence = float(confidence_t.item())

        return DetectionResult(
            box_coordinates=None,
            confidence=confidence,
            class_id=class_id,
            class_name=get_sign_name(class_id),
            is_valid=is_valid_class(class_id),
        )

    # =========================================================================
    # Private helpers — validation
    # =========================================================================

    def _validate(self, class_id: int) -> bool:
        """Return True for valid Zalo foreground classes (1–7)."""
        if not is_valid_class(class_id):
            return False

        # Cross-check against QCVN 41:2019 PDF rules for known sign codes.
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

        return True
