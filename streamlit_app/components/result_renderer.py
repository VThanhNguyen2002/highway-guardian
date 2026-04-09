"""
streamlit_app/components/result_renderer.py
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


def render_results(image: Image.Image, predictions: list[dict[str, Any]], mode: str = "yolo") -> None:
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if mode == "cnn":
        # Do not draw bounding boxes, just show the image
        st.image(image, use_container_width=True)
        st.divider()
    else:
        for idx, pred in enumerate(predictions):
            coords = pred.get("box_coordinates")
            if not coords or len(coords) != 4:
                continue

            x1, y1, x2, y2 = coords
            colour = "green" if pred.get("is_valid", False) else "red"

            draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
            
            # Only draw Class ID on image, NOT full UTF-8 text
            label = f"#{idx+1} (ID:{pred['class_id']})"
            
            if font is not None:
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w, text_h = 50, 15
            else:
                text_w, text_h = 50, 15
                
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=colour)
            
            if font is not None:
                draw.text((x1 + 2, y1 - text_h - 2), label, fill="white", font=font)

        st.image(image, use_container_width=True)

    if not predictions:
        st.warning("No traffic signs detected.")
        return

    st.subheader("Detected Objects" if mode == "yolo" else "Classification Result")
    
    for idx, pred in enumerate(predictions):
        with st.container():
            colour = "🟢" if pred.get("is_valid", False) else "🔴"
            title = f"Sign #{idx+1}" if mode == "yolo" else "MobileNetV2 Classification"
            st.markdown(f"#### {title}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Predicted Class**  \n{colour} {pred['class_name']}")
            with col2:
                st.info(f"**Confidence**  \n{pred['confidence']:.1%}")
            with col3:
                status_text = "Valid (2026/QCVN41)" if pred.get("is_valid", False) else "Invalid"
                if pred.get("is_valid", False):
                    st.success(f"**Compliance**  \n{status_text}")
                else:
                    st.error(f"**Compliance**  \n{status_text}")
            
            st.caption(f"Class ID: {pred['class_id']}")
            st.divider()
