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
from ice_servers import get_ice_servers
from landmarkers import RunningMode, create_face_landmarker, create_hand_landmarker, to_image

PANEL_W, PANEL_H = 480, 360


class FaceGestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_landmarker = create_face_landmarker(running_mode=RunningMode.VIDEO)
        self.hand_landmarker = create_hand_landmarker(running_mode=RunningMode.VIDEO)
        self.smoother = GestureSmoother()

        self._lock = threading.Lock()
        self._last_gesture = "UNKNOWN"
        self._start_time = time.monotonic()
        self._last_ts = -1  # Tracks last timestamp to guarantee strict monotonicity

    @property
    def last_gesture(self):
        with self._lock:
            return self._last_gesture

    def _timestamp_ms(self):
        current_ts = int((time.monotonic() - self._start_time) * 1000)
        if current_ts <= self._last_ts:
            current_ts = self._last_ts + 1
        self._last_ts = current_ts
        return current_ts

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = to_image(rgb)
        ts = self._timestamp_ms()

        face_result = self.face_landmarker.detect_for_video(mp_image, ts)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, ts)

        h, w = img.shape[:2]

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

        face_px = []
        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]
            face_px = [(lm.x * PANEL_W, lm.y * PANEL_H) for lm in face_lms]

        mesh_panel = draw_face_mesh_panel(face_px, color, PANEL_W, PANEL_H)

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

    ice_servers = get_ice_servers()
    rtc_configuration = RTCConfiguration({"iceServers": ice_servers})

    webrtc_streamer(
        key="face-gesture-mesh",
        video_processor_factory=FaceGestureProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )


if __name__ == "__main__":
    main()
