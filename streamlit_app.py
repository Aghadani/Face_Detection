import time

import cv2
import numpy as np
import streamlit as st

from gesture_logic import (
    GESTURE_COLORS,
    GestureSmoother,
    classify_gesture,
    draw_face_mesh_panel,
    fingers_up,
)
from landmarkers import create_face_landmarker, create_hand_landmarker, to_image

PANEL_W, PANEL_H = 480, 360


@st.cache_resource
def load_landmarkers():
    """Loaded once per server process (cached across reruns/sessions).
    Uses IMAGE running mode -- each camera_input capture is an independent
    photo with no temporal relationship to the previous one, so VIDEO
    mode's frame-to-frame tracking assumptions don't apply here."""
    import mediapipe.tasks.python.vision as mp_vision
    face = create_face_landmarker(running_mode=mp_vision.RunningMode.IMAGE)
    hand = create_hand_landmarker(running_mode=mp_vision.RunningMode.IMAGE)
    return face, hand


def process_snapshot(frame_bgr, face_landmarker, hand_landmarker, smoother):
    """Runs detection on a single captured photo and returns the combined
    split-screen image plus the detected gesture name."""
    frame_bgr = cv2.flip(frame_bgr, 1)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = to_image(rgb)

    face_result = face_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)

    h, w = frame_bgr.shape[:2]

    gesture = "UNKNOWN"
    if hand_result.hand_landmarks and hand_result.handedness:
        hand_lms = hand_result.hand_landmarks[0]
        handedness_label = hand_result.handedness[0][0].category_name
        coords = [(lm.x, lm.y, lm.z) for lm in hand_lms]
        state = fingers_up(coords, handedness_label)
        raw_gesture = classify_gesture(state)
        gesture = smoother.update(raw_gesture)

        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 0), -1)
    else:
        gesture = smoother.update("UNKNOWN")

    color = GESTURE_COLORS[gesture]

    face_px = []
    if face_result.face_landmarks:
        face_lms = face_result.face_landmarks[0]
        face_px = [(lm.x * PANEL_W, lm.y * PANEL_H) for lm in face_lms]

    mesh_panel = draw_face_mesh_panel(face_px, color, PANEL_W, PANEL_H)

    left = cv2.resize(frame_bgr, (PANEL_W, PANEL_H))
    cv2.putText(left, f"Gesture: {gesture}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    combined_bgr = np.hstack([left, mesh_panel])
    combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
    return combined_rgb, gesture


def main():
    st.set_page_config(page_title="Face Mesh + Gesture Color", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 8% 5%, rgba(73, 104, 180, 0.10), transparent 28%),
                radial-gradient(circle at 92% 7%, rgba(38, 150, 130, 0.06), transparent 25%),
                #07090d;
            color: #eef2f7;
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            max-width: 1420px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .fm-hero {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 32px;
            padding: 0 2px 24px;
        }

        .fm-eyebrow {
            color: #748196;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .20em;
            text-transform: uppercase;
            margin-bottom: 9px;
        }

        .fm-title {
            color: #f2f5f9;
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(34px, 4vw, 52px);
            font-weight: 700;
            line-height: .98;
            letter-spacing: -.045em;
            margin: 0;
        }

        .fm-subtitle {
            color: #8591a4;
            font-size: 13px;
            line-height: 1.7;
            max-width: 720px;
            margin: 13px 0 0;
        }

        .fm-live {
            border: 1px solid #202934;
            background: rgba(13,17,23,.84);
            border-radius: 999px;
            color: #aeb8c6;
            padding: 10px 15px;
            font-size: 11px;
            white-space: nowrap;
            margin-bottom: 5px;
        }

        .fm-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #5bd9b5;
            box-shadow: 0 0 13px rgba(91,217,181,.65);
            margin-right: 8px;
        }

        .fm-section {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #a0abba;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin: 8px 0 8px;
        }

        .fm-section span:last-child {
            color: #586577;
            font-weight: 500;
            letter-spacing: 0;
            text-transform: none;
        }

        .fm-card {
            background: linear-gradient(145deg, rgba(20,25,33,.97), rgba(10,13,18,.97));
            border: 1px solid #202934;
            border-radius: 16px;
            padding: 15px 17px;
            margin-top: 15px;
        }

        .fm-card-title {
            color: #f0f3f7;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 19px;
            font-weight: 600;
        }

        .fm-card-text {
            color: #737f91;
            font-size: 11px;
            line-height: 1.65;
            margin-top: 5px;
        }

        .fm-legend {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 7px;
            margin-top: 12px;
        }

        .fm-legend-item {
            background: #0b0f15;
            border: 1px solid #1b232e;
            border-radius: 10px;
            padding: 9px 10px;
            color: #8994a5;
            font-size: 10px;
        }

        .fm-color-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            margin-right: 6px;
        }

        div[data-testid="stCameraInput"] {
            border: 1px solid #222b36;
            border-radius: 18px;
            padding: 10px;
            background: #090c11;
            overflow: hidden;
        }

        div[data-testid="stCameraInput"] button {
            background: #151b24 !important;
            color: #edf2f7 !important;
            border: 1px solid #303a48 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            min-height: 42px !important;
        }

        div[data-testid="stCameraInput"] button:hover {
            background: #1c2430 !important;
            border-color: #566476 !important;
        }

        div[data-testid="stCameraInput"] button p,
        div[data-testid="stCameraInput"] button span {
            color: #edf2f7 !important;
        }

        section[data-testid="stSidebar"] {
            background: #080b10;
            border-right: 1px solid #1a212b;
        }

        .fm-footer {
            text-align: center;
            color: #414b59;
            font-size: 9px;
            letter-spacing: .14em;
            text-transform: uppercase;
            padding-top: 25px;
        }

        @media (max-width: 800px) {
            .fm-hero { display: block; }
            .fm-live { display: inline-block; margin-top: 18px; }
            .fm-legend { grid-template-columns: repeat(2, 1fr); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="fm-hero">
          <div>
            <div class="fm-eyebrow">Computer Vision Studio · MediaPipe</div>
            <div class="fm-title">Face Mesh Studio</div>
            <div class="fm-subtitle">
              Gesture-controlled facial landmark visualization. Detect the
              face and hand, then explore the resulting mesh through a clean,
              technical computer-vision interface.
            </div>
          </div>
          <div class="fm-live"><span class="fm-dot"></span>VISION PIPELINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="fm-section"><span>01 · Camera & Analysis</span>'
        '<span>Face + hand landmark detection</span></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            '<div class="fm-eyebrow">Interaction</div>'
            '<div class="fm-card-title">Gesture palette</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="fm-legend">
              <div class="fm-legend-item"><span class="fm-color-dot" style="background:#e23f55"></span>Fist</div>
              <div class="fm-legend-item"><span class="fm-color-dot" style="background:#4fc68d"></span>Open palm</div>
              <div class="fm-legend-item"><span class="fm-color-dot" style="background:#4b84e8"></span>Peace</div>
              <div class="fm-legend-item"><span class="fm-color-dot" style="background:#d8c34b"></span>Thumbs up</div>
              <div class="fm-legend-item"><span class="fm-color-dot" style="background:#c954df"></span>Point</div>
              <div class="fm-legend-item"><span class="fm-color-dot" style="background:#808896"></span>Unknown</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="fm-card-text">Rule-based classification on 2D hand '
            'landmarks. Gesture output is smoothed across recent frames.</div>',
            unsafe_allow_html=True,
        )
        st.caption("Models are downloaded once and cached locally.")

    if "smoother" not in st.session_state:
        st.session_state.smoother = GestureSmoother()

    face_landmarker, hand_landmarker = load_landmarkers()

    img_file = st.camera_input("Camera", label_visibility="visible")

    if img_file is not None:
        file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Detecting face and hand..."):
            combined_rgb, gesture = process_snapshot(
                frame_bgr, face_landmarker, hand_landmarker,
                st.session_state.smoother,
            )

        st.image(combined_rgb, caption=f"Detected gesture: {gesture}",
                  use_container_width=True)


    st.markdown(
        '<div class="fm-footer">FACE MESH STUDIO · COMPUTER VISION INTERFACE</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
