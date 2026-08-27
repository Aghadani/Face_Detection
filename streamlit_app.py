import sys

# --- MONKEY-PATCH FOR PYTHON 3.14 + AIOICE/STUN TEARDOWN ERROR ---
# Must be executed BEFORE importing/running streamlit-webrtc
try:
    import aioice.stun

    def _safe_retry(self):
        try:
            # Access protocol dynamically to handle name mangling safely
            protocol = getattr(self, "_Transaction__protocol", None)
            request = getattr(self, "_Transaction__request", None)
            addr = getattr(self, "_Transaction__addr", None)

            if protocol is not None and getattr(protocol, "transport", None) is not None:
                if getattr(protocol.transport, "_sock", None) is not None:
                    protocol.send_stun(request, addr)
        except Exception:
            # Quietly suppress closed transport/NoneType errors during Streamlit reruns
            pass

    # Replace the retry callback with the safe wrapper
    aioice.stun.Transaction._Transaction__retry = _safe_retry
except Exception:
    pass
# ------------------------------------------------------------------

import threading
import time

import av
import cv2
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
from landmarkers import create_face_landmarker, create_hand_landmarker, to_image

PANEL_W, PANEL_H = 480, 360  # Smaller per-panel size keeps WebRTC bitrate reasonable

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]},
        ]
    }
)


@st.cache_resource
def get_cached_face_landmarker():
    return create_face_landmarker()


@st.cache_resource
def get_cached_hand_landmarker():
    return create_hand_landmarker()


class FaceGestureProcessor(VideoProcessorBase):

    def __init__(self):
        self.face_landmarker = get_cached_face_landmarker()
        self.hand_landmarker = get_cached_hand_landmarker()
        self.smoother = GestureSmoother()

        self._lock = threading.Lock()
        self._last_gesture = "UNKNOWN"
        self._start_time = time.monotonic()

    @property
    def last_gesture(self):
        with self._lock:
            return self._last_gesture

    def _timestamp_ms(self):
        return int((time.monotonic() - self._start_time) * 1000)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = to_image(rgb)
        ts = self._timestamp_ms()

        face_result = self.face_landmarker.detect_for_video(mp_image, ts)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, ts)

        h, w = img.shape[:2]

        # --- Hand gesture ---
        gesture = "UNKNOWN"
        if hand_result.hand_landmarks and hand_result.handedness:
            hand_lms = hand_result.hand_landmarks[0]
            handedness_label = hand_result.handedness[0][0].category_name
            coords = [(lm.x, lm.y, lm.z) for lm in hand_lms]
            state = fingers_up(coords, handedness_label)
            raw_gesture = classify_gesture(state)
            gesture = self.smoother.update(raw_gesture)

            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
        else:
            gesture = self.smoother.update("UNKNOWN")

        with self._lock:
            self._last_gesture = gesture

        color = GESTURE_COLORS[gesture]

        # --- Face mesh panel ---
        face_px = []
        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]
            face_px = [(lm.x * PANEL_W, lm.y * PANEL_H) for lm in face_lms]

        mesh_panel = draw_face_mesh_panel(face_px, color, PANEL_W, PANEL_H)

        left = cv2.resize(img, (PANEL_W, PANEL_H))
        cv2.putText(
            left,
            f"Gesture: {gesture}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        combined = np.hstack([left, mesh_panel])
        return av.VideoFrame.from_ndarray(combined, format="bgr24")

    def close(self):
        with self._lock:
            self._last_gesture = "STOPPED"


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
        async_processing=True,
    )


if __name__ == "__main__":
    main()
