"""
Streamlit app: Face Mesh + Hand Gesture Color Control (deploy-ready)

Uses streamlit-webrtc so the video runs as real browser-side WebRTC
streaming (not a polling loop), suitable for local use or deployment
(e.g. Streamlit Community Cloud, with a STUN/TURN server configured
for NAT traversal in restrictive networks).

Run:
    streamlit run streamlit_app.py
"""

import threading

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from gesture_logic import (
    GESTURE_COLORS,
    GestureSmoother,
    classify_gesture,
    draw_face_mesh_panel,
    fingers_up,
)

PANEL_W, PANEL_H = 480, 360  # smaller per-panel size keeps WebRTC bitrate reasonable

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class FaceGestureProcessor(VideoProcessorBase):
    """Runs face-mesh + hand-gesture detection per frame and returns a
    single combined (split-screen) frame. State that the main Streamlit
    thread reads (current gesture) is guarded by a lock, since this
    recv() callback runs on a separate WebRTC worker thread."""

    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        mp_hands = mp.solutions.hands

        self.mp_hands = mp_hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.connections = list(mp_face_mesh.FACEMESH_TESSELATION)

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.smoother = GestureSmoother()

        self._lock = threading.Lock()
        self._last_gesture = "UNKNOWN"

    @property
    def last_gesture(self):
        with self._lock:
            return self._last_gesture

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb)
        hand_results = self.hands.process(rgb)

        gesture = "UNKNOWN"
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            hand_lms = hand_results.multi_hand_landmarks[0]
            handedness_label = hand_results.multi_handedness[0].classification[0].label
            coords = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
            state = fingers_up(coords, handedness_label)
            raw_gesture = classify_gesture(state)
            gesture = self.smoother.update(raw_gesture)

            self.mp_drawing.draw_landmarks(
                img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
        else:
            gesture = self.smoother.update("UNKNOWN")

        with self._lock:
            self._last_gesture = gesture

        color = GESTURE_COLORS[gesture]

        face_px = None
        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0]
            face_px = [
                (int(lm.x * PANEL_W), int(lm.y * PANEL_H))
                for lm in face_lms.landmark
            ]

        mesh_panel = draw_face_mesh_panel(face_px, color, self.connections, PANEL_W, PANEL_H)

        left = cv2.resize(img, (PANEL_W, PANEL_H))
        cv2.putText(left, f"Gesture: {gesture}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined = np.hstack([left, mesh_panel])
        return av.VideoFrame.from_ndarray(combined, format="bgr24")


def main():
    st.set_page_config(page_title="Face Mesh + Gesture Color", layout="wide")
    st.title("Face Mesh + Hand Gesture Color Control")

    st.markdown(
        "Left: camera feed with hand landmarks. Right: live face mesh, "
        "colored by your hand gesture."
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
            "the weakest point, and peace/point can be confused during "
            "finger transitions. Smoothed over 8 frames to reduce flicker."
        )

    webrtc_streamer(
        key="face-gesture-mesh",
        video_processor_factory=FaceGestureProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )


if __name__ == "__main__":
    main()
