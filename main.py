"""
Face Mesh + Hand Gesture Color Control (desktop / OpenCV window version)
--------------------------------------------------------------------------
Split-screen live app:
  LEFT  : raw webcam feed with hand landmarks + detected gesture label
  RIGHT : black canvas showing the live face mesh wireframe, colored
          according to the current hand gesture

Run:
    python main.py

Requires a webcam. Press 'q' to quit.

For a browser-based / deployable version, see streamlit_app.py instead --
this script uses cv2.imshow(), which only works in a local desktop window.
"""

import cv2
import mediapipe as mp
import numpy as np

from gesture_logic import (
    GESTURE_COLORS,
    GestureSmoother,
    classify_gesture,
    draw_face_mesh_panel,
    fingers_up,
)

PANEL_W, PANEL_H = 640, 480


def main():
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    smoother = GestureSmoother()
    connections = list(mp_face_mesh.FACEMESH_TESSELATION)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). "
                            "Check camera permissions / index.")

    print("Running. Press 'q' in the window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        gesture = "UNKNOWN"
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            hand_lms = hand_results.multi_hand_landmarks[0]
            handedness_label = hand_results.multi_handedness[0].classification[0].label
            coords = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
            state = fingers_up(coords, handedness_label)
            raw_gesture = classify_gesture(state)
            gesture = smoother.update(raw_gesture)

            mp_drawing.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
        else:
            gesture = smoother.update("UNKNOWN")

        color = GESTURE_COLORS[gesture]

        face_px = None
        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0]
            face_px = [(int(lm.x * PANEL_W), int(lm.y * PANEL_H))
                       for lm in face_lms.landmark]

        mesh_panel = draw_face_mesh_panel(face_px, color, connections, PANEL_W, PANEL_H)

        left = cv2.resize(frame, (PANEL_W, PANEL_H))
        cv2.putText(left, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.hstack([left, mesh_panel])
        cv2.imshow("Face Mesh + Gesture Color Control", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()


if __name__ == "__main__":
    main()
