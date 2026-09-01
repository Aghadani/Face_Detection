import time

import cv2
import numpy as np

from gesture_logic import (
    GESTURE_COLORS,
    GestureSmoother,
    classify_gesture,
    draw_face_mesh_panel,
    fingers_up,
)
from landmarkers import create_face_landmarker, create_hand_landmarker, to_image

PANEL_W, PANEL_H = 640, 480


def main():
    print("Loading models (first run downloads them, then caches)...")
    face_landmarker = create_face_landmarker()
    hand_landmarker = create_hand_landmarker()
    smoother = GestureSmoother()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). "
                            "Check camera permissions / index.")

    start_time = time.monotonic()
    print("Running. Press 'q' in the window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = to_image(rgb)
        ts = int((time.monotonic() - start_time) * 1000)

        face_result = face_landmarker.detect_for_video(mp_image, ts)
        hand_result = hand_landmarker.detect_for_video(mp_image, ts)

        h, w = frame.shape[:2]

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
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        else:
            gesture = smoother.update("UNKNOWN")

        color = GESTURE_COLORS[gesture]

        face_px = []
        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]
            face_px = [(lm.x * PANEL_W, lm.y * PANEL_H) for lm in face_lms]

        mesh_panel = draw_face_mesh_panel(face_px, color, PANEL_W, PANEL_H)

        left = cv2.resize(frame, (PANEL_W, PANEL_H))
        cv2.putText(left, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.hstack([left, mesh_panel])
        cv2.imshow("Face Mesh + Gesture Color Control", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    hand_landmarker.close()


if __name__ == "__main__":
    main()
