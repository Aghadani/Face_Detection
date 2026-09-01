import cv2
import numpy as np
import streamlit as st

from gesture_logic import (
    GESTURE_COLORS,
    GestureSmoother,
    classify_gesture,
    fingers_up,
)
from landmarkers import create_face_landmarker, create_hand_landmarker, to_image

FRAME_W, FRAME_H = 720, 540
MESH_W, MESH_H = 720, 540


# ---------- UI ----------
def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(80, 130, 255, .10), transparent 30%),
                radial-gradient(circle at 88% 12%, rgba(0, 220, 190, .07), transparent 28%),
                #07090d;
            color: #eef2f7;
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 2.1rem;
            padding-bottom: 3rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .hero {
            display:flex;
            justify-content:space-between;
            align-items:flex-end;
            gap:30px;
            padding: 4px 2px 20px 2px;
        }

        .eyebrow {
            color:#8290a5;
            font-size:11px;
            font-weight:700;
            letter-spacing:.18em;
            text-transform:uppercase;
            margin-bottom:10px;
        }

        .hero h1 {
            margin:0;
            font-family:'Space Grotesk', sans-serif;
            font-size:clamp(30px, 4vw, 48px);
            line-height:1.02;
            letter-spacing:-.045em;
            color:#f5f7fa;
        }

        .hero p {
            margin:13px 0 0;
            color:#8e99aa;
            max-width:720px;
            font-size:14px;
            line-height:1.65;
        }

        .status-pill {
            border:1px solid #202833;
            background:rgba(17,21,28,.72);
            border-radius:999px;
            padding:9px 14px;
            color:#aeb8c7;
            font-size:12px;
            white-space:nowrap;
        }

        .status-dot {
            display:inline-block;
            width:7px;
            height:7px;
            border-radius:50%;
            background:#57d6b2;
            margin-right:7px;
            box-shadow:0 0 12px rgba(87,214,178,.55);
        }

        .panel-title {
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin:18px 0 8px;
            color:#aeb8c7;
            font-size:11px;
            font-weight:700;
            letter-spacing:.13em;
            text-transform:uppercase;
        }

        .hint {
            color:#606c7d;
            font-weight:500;
            letter-spacing:0;
            text-transform:none;
        }

        .metric-card {
            background:linear-gradient(145deg, rgba(20,25,33,.96), rgba(11,14,19,.96));
            border:1px solid #222a35;
            border-radius:16px;
            padding:15px 17px;
            min-height:72px;
        }

        .metric-label {
            color:#687487;
            font-size:10px;
            font-weight:700;
            letter-spacing:.13em;
            text-transform:uppercase;
        }

        .metric-value {
            color:#f3f6fa;
            font-size:18px;
            font-weight:700;
            margin-top:5px;
        }

        .metric-sub {
            color:#697588;
            font-size:11px;
            margin-top:2px;
        }

        .gesture-card {
            background:linear-gradient(145deg, rgba(21,27,36,.98), rgba(10,13,18,.98));
            border:1px solid #28313e;
            border-radius:18px;
            padding:18px;
            margin-top:12px;
        }

        .gesture-name {
            color:#f4f6f9;
            font-family:'Space Grotesk',sans-serif;
            font-size:25px;
            font-weight:700;
            letter-spacing:-.025em;
        }

        .gesture-desc {
            color:#788496;
            font-size:12px;
            margin-top:5px;
        }

        .color-chip {
            width:42px;
            height:42px;
            border-radius:12px;
            border:1px solid rgba(255,255,255,.13);
            margin-left:auto;
        }

        .legend {
            display:grid;
            grid-template-columns:repeat(3,1fr);
            gap:8px;
            margin-top:10px;
        }

        .legend-item {
            border:1px solid #1d2530;
            background:#0c1016;
            border-radius:12px;
            padding:9px 10px;
            color:#8994a4;
            font-size:11px;
        }

        .legend-dot {
            display:inline-block;
            width:8px;
            height:8px;
            border-radius:50%;
            margin-right:6px;
        }

        div[data-testid="stCameraInput"] {
            border:1px solid #222a35;
            border-radius:18px;
            padding:10px;
            background:#0a0d12;
            overflow:hidden;
        }

        div[data-testid="stCameraInput"] button {
            background:#151b24 !important;
            color:#eef2f7 !important;
            border:1px solid #303a48 !important;
            border-radius:10px !important;
            font-weight:600 !important;
            min-height:42px !important;
        }

        div[data-testid="stCameraInput"] button:hover {
            background:#202936 !important;
            border-color:#566477 !important;
            color:#ffffff !important;
        }

        div[data-testid="stCameraInput"] button p,
        div[data-testid="stCameraInput"] button span {
            color:#eef2f7 !important;
        }

        div[data-testid="stCameraInput"] label {
            color:#7f8b9d !important;
        }

        .footer {
            text-align:center;
            color:#4f5a69;
            font-size:10px;
            letter-spacing:.08em;
            text-transform:uppercase;
            padding-top:25px;
        }

        section[data-testid="stSidebar"] {
            background:#080b10;
            border-right:1px solid #1a212b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def rgb_hex(bgr):
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def color_name(gesture):
    return {
        "FIST": "Red",
        "OPEN_PALM": "Green",
        "PEACE": "Blue",
        "THUMBS_UP": "Yellow",
        "POINT": "Magenta",
        "UNKNOWN": "Neutral",
    }.get(gesture, "Neutral")


# ---------- Processing ----------
@st.cache_resource
def load_landmarkers():
    from mediapipe.tasks.python import vision as mp_vision

    face = create_face_landmarker(running_mode=mp_vision.RunningMode.IMAGE)
    hand = create_hand_landmarker(running_mode=mp_vision.RunningMode.IMAGE)
    return face, hand


def smooth_points(points, previous, alpha=0.72):
    if previous is None or len(previous) != len(points):
        return np.asarray(points, dtype=np.float32)

    current = np.asarray(points, dtype=np.float32)
    old = np.asarray(previous, dtype=np.float32)
    return alpha * old + (1.0 - alpha) * current


def draw_premium_mesh(face_px, color_bgr, width, height, previous_points=None):
    """Build a dark technical mesh canvas with subtle depth and glow."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Very subtle technical-grid background.
    for x in range(0, width, 60):
        cv2.line(canvas, (x, 0), (x, height), (13, 18, 25), 1)
    for y in range(0, height, 60):
        cv2.line(canvas, (0, y), (width, y), (13, 18, 25), 1)

    if not face_px:
        cv2.putText(
            canvas,
            "FACE NOT DETECTED",
            (width // 2 - 105, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 110, 125),
            1,
            cv2.LINE_AA,
        )
        return canvas, previous_points

    pts = smooth_points(face_px, previous_points)
    pts_int = np.rint(pts).astype(np.int32)

    # MediaPipe Face Landmarker provides 478 landmarks. These are the
    # standard tessellation edge indices used for a clean wireframe.
    # A compact deterministic graph keeps the display elegant instead of
    # drawing every possible connection.
    try:
        from mediapipe.tasks.python import vision as mp_vision

        connections = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
        for connection in connections:
            a = connection.start
            b = connection.end
            if a < len(pts_int) and b < len(pts_int):
                cv2.line(canvas, tuple(pts_int[a]), tuple(pts_int[b]),
                         tuple(int(v) for v in color_bgr), 1, cv2.LINE_AA)
    except Exception:
        # Fallback: connect adjacent points if MediaPipe connection metadata
        # is unavailable in the installed version.
        for i in range(len(pts_int) - 1):
            cv2.line(canvas, tuple(pts_int[i]), tuple(pts_int[i + 1]),
                     tuple(int(v) for v in color_bgr), 1, cv2.LINE_AA)

    # Fine points on the mesh.
    for x, y in pts_int[::4]:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (int(x), int(y)), 1, tuple(int(v) for v in color_bgr), -1)

    # Small technical label.
    cv2.putText(canvas, "FACE MESH / 3D LANDMARK FIELD",
                (18, height - 20), cv2.FONT_HERSHEY_SIMPLEX, .42,
                (91, 103, 120), 1, cv2.LINE_AA)

    return canvas, pts


def process_snapshot(frame_bgr, face_landmarker, hand_landmarker, smoother):
    # Mirror the image so the user sees a natural camera view.
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
        gesture = smoother.update(classify_gesture(state))

        # Elegant hand landmark treatment.
        accent = tuple(int(v) for v in GESTURE_COLORS[gesture])
        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 4, accent, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (cx, cy), 7, (245, 247, 250), 1, cv2.LINE_AA)
    else:
        gesture = smoother.update("UNKNOWN")

    return frame_bgr, face_result, gesture


def make_camera_display(frame_bgr, gesture):
    """Clean camera treatment with an understated HUD."""
    display = cv2.resize(frame_bgr, (FRAME_W, FRAME_H))
    accent = tuple(int(v) for v in GESTURE_COLORS[gesture])

    # Dark translucent top strip.
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_W, 56), (5, 8, 12), -1)
    display = cv2.addWeighted(overlay, .72, display, .28, 0)

    cv2.putText(display, "LIVE CAPTURE",
                (18, 25), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (155, 166, 181), 1, cv2.LINE_AA)
    cv2.putText(display, gesture.replace("_", " "),
                (18, 46), cv2.FONT_HERSHEY_SIMPLEX, .62,
                accent, 2, cv2.LINE_AA)

    # Corner brackets.
    c = (185, 194, 207)
    L = 22
    t = 2
    for x, y, sx, sy in [
        (10, 70, 1, 1), (FRAME_W - 10, 70, -1, 1),
        (10, FRAME_H - 12, 1, -1), (FRAME_W - 10, FRAME_H - 12, -1, -1)
    ]:
        cv2.line(display, (x, y), (x + sx * L, y), c, t)
        cv2.line(display, (x, y), (x, y + sy * L), c, t)

    return display


def main():
    st.set_page_config(
        page_title="Face Mesh Studio",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="eyebrow">Computer Vision Studio · MediaPipe</div>
            <h1>Face Mesh Studio</h1>
            <p>
              Gesture-controlled facial landmark visualization.
              Capture a frame, detect the face and hand, then explore the
              resulting mesh in a restrained technical interface.
            </p>
          </div>
          <div class="status-pill"><span class="status-dot"></span>LOCAL VISION PIPELINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "smoother" not in st.session_state:
        st.session_state.smoother = GestureSmoother()
    if "mesh_points" not in st.session_state:
        st.session_state.mesh_points = None

    face_landmarker, hand_landmarker = load_landmarkers()

    # Capture and visualization are intentionally kept on one screen.
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            '<div class="panel-title"><span>01 · Camera</span><span class="hint">Capture a frame</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Position your face in frame, then select **Take Photo** to run the vision pipeline.")
        img_file = st.camera_input("Camera capture", label_visibility="collapsed")

    gesture = "UNKNOWN"
    face_result = None

    if img_file is not None:
        file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Analyzing facial and hand landmarks..."):
            camera_frame, face_result, gesture = process_snapshot(
                frame_bgr,
                face_landmarker,
                hand_landmarker,
                st.session_state.smoother,
            )

        with left:
            st.image(
                cv2.cvtColor(make_camera_display(camera_frame, gesture), cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        face_px = []
        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]
            face_px = [
                (lm.x * MESH_W, lm.y * MESH_H)
                for lm in face_lms
            ]

        mesh_bgr, st.session_state.mesh_points = draw_premium_mesh(
            face_px,
            GESTURE_COLORS[gesture],
            MESH_W,
            MESH_H,
            st.session_state.mesh_points,
        )

        with right:
            st.markdown(
                '<div class="panel-title"><span>02 · Mesh Field</span><span class="hint">Gesture-responsive</span></div>',
                unsafe_allow_html=True,
            )
            st.image(
                cv2.cvtColor(mesh_bgr, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

    else:
        with left:
            st.markdown(
                '<div class="panel-title"><span>01 · Camera</span><span class="hint">Awaiting capture</span></div>',
                unsafe_allow_html=True,
            )
            st.info("Allow camera access and capture a frame to begin.")

        with right:
            st.markdown(
                '<div class="panel-title"><span>02 · Mesh Field</span><span class="hint">Awaiting capture</span></div>',
                unsafe_allow_html=True,
            )
            empty = np.zeros((MESH_H, MESH_W, 3), dtype=np.uint8)
            for x in range(0, MESH_W, 60):
                cv2.line(empty, (x, 0), (x, MESH_H), (13, 18, 25), 1)
            for y in range(0, MESH_H, 60):
                cv2.line(empty, (0, y), (MESH_W, y), (13, 18, 25), 1)
            cv2.putText(
                empty, "CAPTURE TO INITIALIZE MESH",
                (MESH_W // 2 - 95, MESH_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (85, 97, 113), 1, cv2.LINE_AA
            )
            st.image(cv2.cvtColor(empty, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Compact status row.
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Gesture</div>'
            f'<div class="metric-value">{gesture.replace("_", " ").title()}</div>'
            f'<div class="metric-sub">Hand classifier output</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        detected = "Detected" if img_file is not None and face_result is not None and face_result.face_landmarks else "Waiting"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Face</div>'
            f'<div class="metric-value">{detected}</div>'
            f'<div class="metric-sub">Single-face tracking</div></div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Mesh</div>'
            f'<div class="metric-value">478 points</div>'
            f'<div class="metric-sub">MediaPipe landmark field</div></div>',
            unsafe_allow_html=True,
        )

    with c4:
        color_label = color_name(gesture)
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Accent</div>'
            f'<div class="metric-value">{color_label}</div>'
            f'<div class="metric-sub">Gesture-controlled mesh</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="gesture-card">
          <div style="display:flex;align-items:center;">
            <div>
              <div class="eyebrow" style="margin-bottom:4px;">Current interaction</div>
              <div class="gesture-name">{gesture.replace("_", " ").title()}</div>
              <div class="gesture-desc">
                The detected hand gesture controls the visual accent of the facial mesh.
              </div>
            </div>
            <div class="color-chip" style="background:{rgb_hex(GESTURE_COLORS[gesture])};"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="legend">
          <div class="legend-item"><span class="legend-dot" style="background:#ff4d5e"></span>Fist · Red</div>
          <div class="legend-item"><span class="legend-dot" style="background:#4fd49b"></span>Open Palm · Green</div>
          <div class="legend-item"><span class="legend-dot" style="background:#4d8dff"></span>Peace · Blue</div>
          <div class="legend-item"><span class="legend-dot" style="background:#e7c84a"></span>Thumbs Up · Yellow</div>
          <div class="legend-item"><span class="legend-dot" style="background:#d85bff"></span>Point · Magenta</div>
          <div class="legend-item"><span class="legend-dot" style="background:#7d8796"></span>Unknown · Neutral</div>
        </div>
        <div class="footer">FACE MESH STUDIO · COMPUTER VISION INTERFACE</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
