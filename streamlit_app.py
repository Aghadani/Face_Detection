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
    
    import mediapipe.tasks.python.vision as mp_vision
    face = create_face_landmarker(running_mode=mp_vision.RunningMode.IMAGE)
    hand = create_hand_landmarker(running_mode=mp_vision.RunningMode.IMAGE)
    return face, hand


def process_snapshot(frame_bgr, face_landmarker, hand_landmarker, smoother):
   
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
    st.title("Face Mesh + Hand Gesture Color Control")

    st.markdown(
        "Take a photo below. Left: your photo with hand landmarks. "
        "Right: face mesh, colored by your hand gesture. "
        "(Snapshot-based -- click **Take Photo** again for a new frame.)"
    )

    with st.sidebar:
        st.subheader("Gesture -> Color")
        st.markdown(
            "- **Fist** -> Red\n"
            "- **Open palm** -> Green\n"
            "- **Peace** -> Blue\n"
            "- **Thumbs up** -> Yellow\n"
            "- **Point** -> Magenta\n"
            "- anything else -> Gray"
        )
        st.caption(
            "Rule-based classifier on 2D landmarks -- thumb detection is "
            "the weakest point, and peace/point can be confused. Each "
            "photo is classified independently (no smoothing across "
            "separate snapshots)."
        )
        st.caption(
            "First load downloads two small model files -- may take a "
            "few seconds."
        )

    if "smoother" not in st.session_state:
        st.session_state.smoother = GestureSmoother()

    face_landmarker, hand_landmarker = load_landmarkers()

    img_file = st.camera_input("Take a photo")

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


if __name__ == "__main__":
    main()
