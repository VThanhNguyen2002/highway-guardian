"""
streamlit_app/app.py

Highway Guardian — Streamlit UI (Image Upload + Live Camera).
"""

from __future__ import annotations

from typing import Any
import time

import streamlit as st
from PIL import Image

import api_client
from components.result_renderer import render_results
from components.camera_feed import render_camera_feed

st.set_page_config(
    page_title="Highway Guardian",
    page_icon="🛑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return a config dict."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
        st.title("Highway Guardian")
        st.caption("AI Traffic Sign Inspector")
        st.divider()

        # ── Input mode ───────────────────────────────────────────────────────
        st.subheader("📡 Input Mode")
        input_mode = st.radio(
            "Select source:",
            options=["📸 Image Upload", "🎥 Live Camera"],
            key="input_mode",
        )

        st.divider()

        # ── Model selection ──────────────────────────────────────────────────
        st.subheader("🧪 Detection Model")
        model_choice = st.radio(
            "Active model:",
            options=["YOLOv8 (Detection)", "MobileNetV2 (Classification)", "Ensemble"],
            key="model_choice",
        )
        model_mode_map = {
            "YOLOv8 (Detection)":        "yolo",
            "MobileNetV2 (Classification)": "cnn",
            "Ensemble":                  "ensemble",
        }
        model_mode = model_mode_map[model_choice]

        st.divider()

        # ── Threshold ────────────────────────────────────────────────────────
        st.subheader("⚙️ Settings")
        confidence = st.slider(
            "YOLO Confidence Threshold",
            min_value=0.10, max_value=0.95, value=0.25, step=0.05,
            disabled=(model_mode == "cnn"),
            help="Minimum confidence for YOLO detection.",
        )

        st.divider()

        # ── Backend URL ──────────────────────────────────────────────────────
        st.markdown("**Backend**")
        backend_url = st.text_input(
            "API URL", value=api_client.BACKEND_URL, key="backend_url_input"
        )
        if backend_url != api_client.BACKEND_URL:
            api_client.BACKEND_URL = backend_url

    return {
        "input_mode":  input_mode,
        "model_mode":  model_mode,
        "confidence":  confidence,
    }


# ---------------------------------------------------------------------------
# Image Upload tab
# ---------------------------------------------------------------------------

def _render_image_tab(confidence: float, mode: str) -> None:
    st.subheader("📸 Upload Image for Detection")

    if mode == "cnn":
        st.info("⚠️ **MobileNetV2 Mode**: Upload a tightly cropped image of a **single** traffic sign.")
    else:
        st.info("Upload a street photo — YOLO will locate all traffic signs.")

    uploaded_file = st.file_uploader(
        "Choose a JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )
    if uploaded_file is None:
        return

    file_bytes = uploaded_file.read()
    col_orig, col_result = st.columns(2)

    with col_orig:
        st.markdown("**Original**")
        uploaded_file.seek(0)
        original = Image.open(uploaded_file).convert("RGB")
        st.image(original, use_container_width=True)

    with col_result:
        st.markdown("**Detection Result**")
        with st.spinner("Submitting image…"):
            init_response: dict[str, Any] = api_client.submit_detect_task(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                confidence_threshold=confidence,
                mode=mode,
            )
        task_id = init_response.get("task_id")
        if not task_id:
            st.error(f"Failed to submit: {init_response.get('error', 'Unknown error')}")
            return

        with st.spinner("Running inference…"):
            for _ in range(60):       # 60 s max
                time.sleep(1)
                resp: dict[str, Any] = api_client.get_detect_task_status(task_id)
                status = resp.get("status")
                if status == "COMPLETED":
                    render_results(original, resp.get("result", []), mode)
                    break
                elif status == "FAILED":
                    st.error(f"Inference error: {resp.get('error', 'Unknown')}")
                    break
                elif not status:
                    st.error(f"Polling error: {resp.get('error', 'Unknown')}")
                    break


# ---------------------------------------------------------------------------
# History tab
# ---------------------------------------------------------------------------

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
                "is_valid":   st.column_config.CheckboxColumn("Valid"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "model_used": st.column_config.TextColumn("Model"),
            },
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _render_sidebar()
    st.title("🛑 Highway Guardian — Traffic Sign Inspector")

    if "📸 Image Upload" in cfg["input_mode"]:
        tab_detect, tab_history = st.tabs(["🔍 Detect", "📜 History"])
        with tab_detect:
            _render_image_tab(cfg["confidence"], cfg["model_mode"])
        with tab_history:
            _render_history_tab()

    else:  # 🎥 Live Camera
        render_camera_feed(cfg["confidence"], cfg["model_mode"])


if __name__ == "__main__":
    main()
