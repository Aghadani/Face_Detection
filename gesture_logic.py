"""
Shared logic for gesture classification and face-mesh rendering,
used by both main.py (OpenCV desktop app) and streamlit_app.py
(Streamlit + streamlit-webrtc app).
"""

import collections

import cv2
import numpy as np

# Gesture -> BGR color for the mesh
GESTURE_COLORS = {
    "FIST": (60, 60, 220),        # red
    "OPEN_PALM": (80, 220, 80),   # green
    "PEACE": (220, 160, 40),      # blue-ish
    "THUMBS_UP": (40, 220, 220),  # yellow
    "POINT": (220, 60, 220),      # magenta
    "UNKNOWN": (160, 160, 160),   # gray (default / no confident gesture)
}

SMOOTHING_WINDOW = 8  # frames of majority-vote smoothing for gesture stability

# Landmark indices (MediaPipe Hands)
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}


def fingers_up(landmarks, handedness_label):
    """
    Returns a dict of {finger_name: bool} for thumb/index/middle/ring/pinky.
    landmarks: list of (x, y, z) normalized coords from MediaPipe.
    handedness_label: 'Left' or 'Right' as reported by MediaPipe (note:
        MediaPipe reports handedness from the camera's perspective, i.e.
        mirrored, since most webcam feeds are used as a mirror).
    """
    state = {}

    # Thumb: compare x of tip vs mcp; direction depends on which hand
    # (this is the classic weak point of rule-based thumb detection --
    # it degrades if the hand is rotated out of a frontal-ish pose)
    if handedness_label == "Right":
        state["thumb"] = landmarks[THUMB_TIP][0] < landmarks[THUMB_MCP][0]
    else:
        state["thumb"] = landmarks[THUMB_TIP][0] > landmarks[THUMB_MCP][0]

    # Other four fingers: up if tip is above (smaller y) than its PIP joint
    for name, tip_idx in FINGER_TIPS.items():
        pip_idx = FINGER_PIPS[name]
        state[name] = landmarks[tip_idx][1] < landmarks[pip_idx][1]

    return state


def classify_gesture(finger_state):
    """Map a finger up/down pattern to a named gesture. Returns 'UNKNOWN'
    if the pattern doesn't cleanly match one of the defined gestures."""
    thumb, index, middle, ring, pinky = (
        finger_state["thumb"],
        finger_state["index"],
        finger_state["middle"],
        finger_state["ring"],
        finger_state["pinky"],
    )

    if not any([thumb, index, middle, ring, pinky]):
        return "FIST"
    if all([thumb, index, middle, ring, pinky]):
        return "OPEN_PALM"
    if index and middle and not ring and not pinky and not thumb:
        return "PEACE"
    if thumb and not any([index, middle, ring, pinky]):
        return "THUMBS_UP"
    if index and not any([middle, ring, pinky, thumb]):
        return "POINT"
    return "UNKNOWN"


class GestureSmoother:
    """Majority-vote smoothing over the last N frames to reduce flicker
    from single-frame misclassifications. This does not fix ambiguous
    poses -- it only suppresses one-frame noise."""

    def __init__(self, window=SMOOTHING_WINDOW):
        self.buf = collections.deque(maxlen=window)

    def update(self, gesture):
        self.buf.append(gesture)
        counts = collections.Counter(self.buf)
        return counts.most_common(1)[0][0]


def compute_delaunay_edges(points_px, panel_w, panel_h):
    """Compute a live Delaunay triangulation of 2D points and return the
    triangle edges as (pt1, pt2) pairs. This replaces the old fixed
    FACEMESH_TESSELATION connection list, which came from
    mediapipe.solutions -- a module that no longer exists in current
    mediapipe releases (Tasks API only, no shipped connection data)."""
    rect = (0, 0, panel_w, panel_h)
    subdiv = cv2.Subdiv2D(rect)
    for (x, y) in points_px:
        cx = min(max(int(x), 0), panel_w - 1)
        cy = min(max(int(y), 0), panel_h - 1)
        subdiv.insert((float(cx), float(cy)))
    triangles = subdiv.getTriangleList()
    edges = []
    for t in triangles:
        pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
        edges.append((pt1, pt2))
        edges.append((pt2, pt3))
        edges.append((pt3, pt1))
    return edges


def draw_face_mesh_panel(face_landmarks_px, color, panel_w, panel_h):
    """Draw a live-triangulated face mesh wireframe on a black canvas."""
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    if not face_landmarks_px:
        cv2.putText(canvas, "No face detected",
                    (max(10, panel_w // 2 - 100), panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        return canvas

    edges = compute_delaunay_edges(face_landmarks_px, panel_w, panel_h)
    for pt1, pt2 in edges:
        cv2.line(canvas,
                  (int(pt1[0]), int(pt1[1])),
                  (int(pt2[0]), int(pt2[1])),
                  color, 1, cv2.LINE_AA)

    return canvas
