"""
streamlit_app/app.py

Highway Guardian — Inference UI (Streamlit).
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from PIL import Image

import api_client
from components.result_renderer import render_results

st.set_page_config(
    page_title="Highway Guardian — A/B Testing",
    page_icon="🛑",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _render_sidebar() -> tuple[float, str]:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/traffic-light.png",
            width=64,
        )
        st.title("Highway Guardian")
        st.caption("AI Traffic Sign Inspector")
        st.divider()

        st.subheader("🧪 A/B Testing Mode")
        mode_choice = st.radio(
            "Select Model to Test:",
            options=["YOLOv8 (Detection)", "MobileNetV2 (Classification)"],
        )
        mode = "yolo" if "YOLO" in mode_choice else "cnn"

        st.divider()
        st.subheader("⚙️ Settings")
        confidence = st.slider(
            label="YOLO Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.25,
            step=0.05,
            disabled=(mode == "cnn"),
            help="Minimum confidence for YOLO detection.",
        )

        st.divider()
        st.markdown("**Backend**")
        backend_url = st.text_input(
            "API URL",
            value=api_client.BACKEND_URL,
            key="backend_url_input",
        )
        if backend_url != api_client.BACKEND_URL:
            api_client.BACKEND_URL = backend_url

    return confidence, mode

def _render_image_tab(confidence: float, mode: str) -> None:
    st.subheader("📸 Upload Image")
    
    if mode == "cnn":
        st.info("⚠️ **MobileNetV2 Mode**: Please upload a tightly cropped image of a SINGLE traffic sign.")
    else:
        st.info("Upload a street image to detect traffic signs.")

    uploaded_file = st.file_uploader(
        "Choose a JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_file is None:
        return

    col_orig, col_result = st.columns(2)
    file_bytes = uploaded_file.read()

    with col_orig:
        st.markdown("**Original Image**")
        original = Image.open(uploaded_file).convert("RGB")
        uploaded_file.seek(0)
        st.image(original, use_container_width=True)

    with col_result:
        st.markdown("**Detection Result**")
        with st.spinner("Running inference..."):
            response: dict[str, Any] = api_client.detect(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                confidence_threshold=confidence,
                mode=mode,
            )

        if not response.get("success"):
            st.error(f"Inference error: {response.get('error', 'Unknown error')}")
            return

        predictions = response.get("predictions", [])
        render_results(original, predictions, mode)

def _render_history_tab() -> None:
    st.subheader("📜 Detection History")
    col_filter, col_limit = st.columns([2, 1])
    with col_filter:
        validity_filter = st.selectbox(
            "Filter by validity",
            options=["All", "Valid only", "Invalid only"],
            key="hist_validity",
        )
    with col_limit:
        limit = st.number_input(
            "Records to load", min_value=5, max_value=200, value=20, step=5
        )

    is_valid_param = None
    if validity_filter == "Valid only":
        is_valid_param = True
    elif validity_filter == "Invalid only":
        is_valid_param = False

    if st.button("🔄 Refresh History"):
        data = api_client.get_history(limit=int(limit), is_valid=is_valid_param)
        records = data.get("records", [])

        if not records:
            st.info("No history records found.")
            return

        st.dataframe(
            data=records,
            use_container_width=True,
            column_config={
                "is_valid": st.column_config.CheckboxColumn("Valid"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

def main() -> None:
    confidence, mode = _render_sidebar()
    st.title("🛑 Highway Guardian — A/B Testing")
    
    tab_detect, tab_history = st.tabs(["🔍 Test", "📜 History"])
    with tab_detect:
        _render_image_tab(confidence, mode)
    with tab_history:
        _render_history_tab()

if __name__ == "__main__":
    main()
