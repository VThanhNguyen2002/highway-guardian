"""
streamlit_app/components/camera_feed.py

Real-time webcam / RTSP inference feed for Highway Guardian.
Uses a Streamlit st.empty() loop to display annotated frames.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import api_client
from components.result_renderer import render_results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _annotate_frame_fast(frame: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes directly on a cv2 BGR frame (no PIL round-trip)."""
    for pred in predictions:
        coords = pred.get("box_coordinates")
        if not coords or len(coords) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in coords)
        color = (50, 205, 50) if pred.get("is_valid", False) else (50, 50, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{pred.get('class_name', '?')} {pred.get('confidence', 0):.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1 - 8, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _submit_and_poll(frame: np.ndarray, confidence: float, model_mode: str) -> list[dict]:
    """Encode frame → submit to backend → poll → return predictions."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return []
    file_bytes = buf.tobytes()

    try:
        init = api_client.submit_detect_task(
            file_bytes=file_bytes,
            filename="frame.jpg",
            confidence_threshold=confidence,
            mode=model_mode,
        )
    except Exception as exc:
        st.warning(f"Submit failed: {exc}")
        return []

    task_id = init.get("task_id")
    if not task_id:
        return []

    for _ in range(15):           # max ~15 s polling
        time.sleep(1.0)
        try:
            resp = api_client.get_detect_task_status(task_id)
        except Exception:
            return []
        if resp.get("status") == "COMPLETED":
            return resp.get("result", [])
        if resp.get("status") == "FAILED":
            return []
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_camera_feed(confidence: float, model_mode: str) -> None:
    """Render the live camera / RTSP inference feed section."""
    st.subheader("📷 Real-time Video Feed")

    # ── Source config ────────────────────────────────────────────────────────
    col_src, col_fps = st.columns([3, 1])
    with col_src:
        source_type = st.radio(
            "Video source",
            ["Webcam (device 0)", "RTSP / IP Camera URL"],
            horizontal=True,
            key="cam_source_type",
        )
    with col_fps:
        fps_limit = st.slider("Max FPS", min_value=1, max_value=5, value=2,
                              key="cam_fps", help="Frames analysed per second (controls server load).")

    rtsp_url = ""
    if source_type == "RTSP / IP Camera URL":
        rtsp_url = st.text_input(
            "RTSP URL",
            placeholder="rtsp://admin:password@192.168.1.100/stream",
            key="cam_rtsp_url",
        )

    # ── Controls ─────────────────────────────────────────────────────────────
    col_start, col_stop = st.columns(2)
    with col_start:
        start = st.button("▶ Start Feed", key="cam_start", type="primary",
                          disabled=st.session_state.get("cam_running", False))
    with col_stop:
        stop = st.button("⏹ Stop Feed", key="cam_stop",
                         disabled=not st.session_state.get("cam_running", False))

    if stop:
        st.session_state["cam_running"] = False
        st.rerun()

    if start:
        st.session_state["cam_running"] = True

    if not st.session_state.get("cam_running", False):
        st.info("Press **▶ Start Feed** to begin real-time detection.")
        return

    # ── Open capture ─────────────────────────────────────────────────────────
    source = rtsp_url if rtsp_url else 0
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.error(
            "❌ Cannot open video source. "
            "Check that your webcam is connected or the RTSP URL is reachable."
        )
        st.session_state["cam_running"] = False
        return

    frame_placeholder = st.empty()
    results_placeholder = st.empty()
    interval = 1.0 / fps_limit

    st.info("🟢 Feed running — press **⏹ Stop Feed** to halt.")

    frame_count = 0
    try:
        while st.session_state.get("cam_running", False):
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Lost connection to video source.")
                break

            frame_count += 1

            # Run inference every frame (FPS already throttled by interval sleep)
            t0 = time.monotonic()
            predictions = _submit_and_poll(frame, confidence, model_mode)
            annotated = _annotate_frame_fast(frame.copy(), predictions)

            # Display annotated frame (BGR → RGB for st.image)
            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=f"Frame #{frame_count} | {len(predictions)} sign(s) detected",
            )

            # Compact result summary below frame
            if predictions:
                with results_placeholder.container():
                    cols = st.columns(min(len(predictions), 4))
                    for i, pred in enumerate(predictions[:4]):
                        with cols[i]:
                            icon = "✅" if pred.get("is_valid") else "❌"
                            conf_pct = f"{pred.get('confidence', 0):.0%}"
                            st.metric(pred.get("class_name", "?"), conf_pct, icon)
            else:
                results_placeholder.empty()

            # Throttle to configured FPS
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

    finally:
        cap.release()
        st.session_state["cam_running"] = False
