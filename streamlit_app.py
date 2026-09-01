import time
import threading

import av
import cv2
import numpy as np
import streamlit as st
from aiortc import RTCConfiguration, RTCIceServer
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

from gesture_logic import (
    GESTURE_COLORS,
    GestureSmoother,
    classify_gesture,
    fingers_up,
)
from landmarkers import create_face_landmarker, create_hand_landmarker, to_image
from ice_servers_updated import get_ice_servers


# ---------------------------------------------------------------------------
# Visual configuration
# ---------------------------------------------------------------------------
OUTPUT_W = 1280
OUTPUT_H = 720
PANEL_W = OUTPUT_W // 2
PANEL_H = OUTPUT_H

# BGR colors used only for the neutral interface / mesh background.
BG = (7, 10, 15)
GRID = (16, 21, 29)
TEXT = (190, 199, 211)
MUTED = (91, 103, 119)
WHITE = (242, 245, 249)


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(65, 105, 190, .10), transparent 28%),
                radial-gradient(circle at 90% 8%, rgba(0, 190, 165, .06), transparent 25%),
                #07090d;
            color: #eef2f7;
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
        }

        .hero {
            display:flex;
            justify-content:space-between;
            align-items:flex-end;
            gap:30px;
            margin-bottom:18px;
        }

        .eyebrow {
            color:#7f8b9e;
            font-size:10px;
            font-weight:700;
            letter-spacing:.19em;
            text-transform:uppercase;
            margin-bottom:8px;
        }

        .hero h1 {
            margin:0;
            font-family:'Space Grotesk',sans-serif;
            font-size:clamp(32px,4vw,48px);
            line-height:1;
            letter-spacing:-.045em;
            color:#f4f6fa;
        }

        .hero p {
            color:#8490a2;
            max-width:760px;
            margin:11px 0 0;
            font-size:13px;
            line-height:1.6;
        }

        .live-pill {
            border:1px solid #202934;
            background:rgba(12,16,22,.82);
            border-radius:999px;
            padding:9px 14px;
            color:#aab5c4;
            font-size:11px;
            white-space:nowrap;
        }

        .live-dot {
            display:inline-block;
            width:7px;
            height:7px;
            border-radius:50%;
            background:#5fe0ba;
            margin-right:7px;
            box-shadow:0 0 12px rgba(95,224,186,.7);
        }

        .section-label {
            display:flex;
            justify-content:space-between;
            margin:9px 0 7px;
            color:#9ba7b7;
            font-size:10px;
            font-weight:700;
            letter-spacing:.15em;
            text-transform:uppercase;
        }

        .section-label span:last-child {
            color:#596576;
            font-weight:500;
            letter-spacing:0;
            text-transform:none;
        }

        .metric {
            background:linear-gradient(145deg,rgba(20,25,33,.96),rgba(10,13,18,.96));
            border:1px solid #202934;
            border-radius:14px;
            padding:13px 15px;
            min-height:66px;
        }

        .metric-k {
            color:#657184;
            font-size:9px;
            font-weight:700;
            letter-spacing:.13em;
            text-transform:uppercase;
        }

        .metric-v {
            color:#f1f4f8;
            font-size:16px;
            font-weight:700;
            margin-top:4px;
        }

        .metric-s {
            color:#687487;
            font-size:10px;
            margin-top:2px;
        }

        .stButton button {
            border:1px solid #293442 !important;
            background:#10151d !important;
            color:#dfe5ed !important;
            border-radius:10px !important;
        }

        .stButton button:hover {
            border-color:#566476 !important;
            background:#171e28 !important;
        }

        footer { visibility:hidden; }

        [data-testid="stWebRtcStreamer"] {
            border-radius:18px;
            overflow:hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# ICE configuration
# ---------------------------------------------------------------------------
@st.cache_resource
def build_rtc_configuration():
    """Convert the Metered ICE-server dictionaries into aiortc objects."""
    raw_servers = get_ice_servers()
    servers = []

    for server in raw_servers:
        if not isinstance(server, dict):
            continue

        urls = server.get("urls")
        if not urls:
            continue

        kwargs = {"urls": urls}

        if server.get("username") is not None:
            kwargs["username"] = server["username"]

        if server.get("credential") is not None:
            kwargs["credential"] = server["credential"]

        servers.append(RTCIceServer(**kwargs))

    # Always keep at least a STUN path if the credential endpoint returns
    # nothing usable.
    if not servers:
        servers = [
            RTCIceServer(urls="stun:stun.l.google.com:19302")
        ]

    return RTCConfiguration(iceServers=servers)


# ---------------------------------------------------------------------------
# Live video processor
# ---------------------------------------------------------------------------
class FaceMeshProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()

        self.face_landmarker = create_face_landmarker()
        self.hand_landmarker = create_hand_landmarker()

        self.smoother = GestureSmoother()
        self.last_timestamp_ms = 0

        self.gesture = "UNKNOWN"
        self.face_detected = False
        self.hand_detected = False
        self.face_points = None
        self.frame_count = 0

    def _timestamp(self):
        now = time.monotonic_ns() // 1_000_000
        if now <= self.last_timestamp_ms:
            now = self.last_timestamp_ms + 1
        self.last_timestamp_ms = now
        return now

    def _gesture(self, result):
        if result.hand_landmarks and result.handedness:
            hand_lms = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name

            coords = [(lm.x, lm.y, lm.z) for lm in hand_lms]
            state = fingers_up(coords, handedness)
            return self.smoother.update(classify_gesture(state))

        return self.smoother.update("UNKNOWN")

    @staticmethod
    def _draw_hand(frame, result, gesture):
        if not result.hand_landmarks:
            return

        h, w = frame.shape[:2]
        accent = tuple(int(v) for v in GESTURE_COLORS[gesture])

        for hand_lms in result.hand_landmarks:
            for lm in hand_lms:
                x = int(lm.x * w)
                y = int(lm.y * h)

                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 3, accent, -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 6, (235, 239, 245), 1, cv2.LINE_AA)

    @staticmethod
    def _technical_background(width, height):
        canvas = np.full((height, width, 3), BG, dtype=np.uint8)

        # Fine grid.
        for x in range(0, width, 48):
            cv2.line(canvas, (x, 0), (x, height), GRID, 1)
        for y in range(0, height, 48):
            cv2.line(canvas, (0, y), (width, y), GRID, 1)

        # Larger guide grid.
        for x in range(0, width, 240):
            cv2.line(canvas, (x, 0), (x, height), (22, 28, 37), 1)
        for y in range(0, height, 240):
            cv2.line(canvas, (0, y), (width, y), (22, 28, 37), 1)

        return canvas

    @staticmethod
    def _draw_mesh(points, color, width, height):
        canvas = FaceMeshProcessor._technical_background(width, height)

        if points is None or len(points) < 20:
            cv2.putText(
                canvas,
                "SEARCHING FOR FACE",
                (width // 2 - 95, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                .55,
                MUTED,
                1,
                cv2.LINE_AA,
            )
            return canvas

        pts = np.asarray(points, dtype=np.float32)
        valid = (
            (pts[:, 0] >= 0) & (pts[:, 0] < width) &
            (pts[:, 1] >= 0) & (pts[:, 1] < height)
        )
        pts = pts[valid]

        if len(pts) < 20:
            return canvas

        # OpenCV Subdiv2D gives us a clean triangulated wireframe without
        # depending on the removed legacy mediapipe.solutions API.
        subdiv = cv2.Subdiv2D((0, 0, width, height))
        for x, y in pts:
            try:
                subdiv.insert((float(x), float(y)))
            except cv2.error:
                pass

        triangle_list = subdiv.getTriangleList()
        accent = tuple(int(v) for v in color)

        for tri in triangle_list:
            x1, y1, x2, y2, x3, y3 = tri

            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            p3 = (int(round(x3)), int(round(y3)))

            if (
                0 <= p1[0] < width and 0 <= p1[1] < height and
                0 <= p2[0] < width and 0 <= p2[1] < height and
                0 <= p3[0] < width and 0 <= p3[1] < height
            ):
                cv2.line(canvas, p1, p2, accent, 1, cv2.LINE_AA)
                cv2.line(canvas, p2, p3, accent, 1, cv2.LINE_AA)
                cv2.line(canvas, p3, p1, accent, 1, cv2.LINE_AA)

        # Small landmark nodes.
        for x, y in pts[::5]:
            cv2.circle(
                canvas,
                (int(x), int(y)),
                1,
                accent,
                -1,
                cv2.LINE_AA,
            )

        # Technical HUD.
        cv2.putText(
            canvas,
            "FACE MESH / LIVE LANDMARK FIELD",
            (18, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            .40,
            TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{len(pts):03d} LANDMARKS",
            (width - 115, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            .34,
            MUTED,
            1,
            cv2.LINE_AA,
        )

        return canvas

    @staticmethod
    def _camera_hud(frame, gesture, fps_text="LIVE"):
        h, w = frame.shape[:2]
        accent = tuple(int(v) for v in GESTURE_COLORS[gesture])

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 62), (5, 8, 12), -1)
        frame[:] = cv2.addWeighted(overlay, .72, frame, .28, 0)

        cv2.putText(
            frame,
            "LIVE CAMERA",
            (18, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            .38,
            TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            gesture.replace("_", " "),
            (18, 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            .58,
            accent,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            fps_text,
            (w - 58, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            .34,
            (100, 218, 185),
            1,
            cv2.LINE_AA,
        )

        # Viewfinder brackets.
        c = (182, 192, 205)
        L = 25
        t = 2
        margin = 12

        corners = [
            (margin, 78, 1, 1),
            (w - margin, 78, -1, 1),
            (margin, h - 14, 1, -1),
            (w - margin, h - 14, -1, -1),
        ]

        for x, y, sx, sy in corners:
            cv2.line(frame, (x, y), (x + sx * L, y), c, t)
            cv2.line(frame, (x, y), (x, y + sy * L), c, t)

        return frame

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Mirror the browser camera for a natural experience.
        image = cv2.flip(image, 1)

        # Normalize processing size. The output remains 16:9.
        camera = cv2.resize(image, (PANEL_W, PANEL_H))

        rgb = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        mp_image = to_image(rgb)

        timestamp = self._timestamp()

        try:
            face_result = self.face_landmarker.detect_for_video(
                mp_image,
                timestamp,
            )
            hand_result = self.hand_landmarker.detect_for_video(
                mp_image,
                timestamp,
            )

            gesture = self._gesture(hand_result)

            face_points = None
            if face_result.face_landmarks:
                face_lms = face_result.face_landmarks[0]
                face_points = np.asarray(
                    [
                        (lm.x * PANEL_W, lm.y * PANEL_H)
                        for lm in face_lms
                    ],
                    dtype=np.float32,
                )

            self.face_detected = bool(face_points is not None)
            self.hand_detected = bool(hand_result.hand_landmarks)
            self.gesture = gesture
            self.face_points = face_points
            self.frame_count += 1

            self._draw_hand(camera, hand_result, gesture)
            camera = self._camera_hud(camera, gesture)

            mesh = self._draw_mesh(
                face_points,
                GESTURE_COLORS[gesture],
                PANEL_W,
                PANEL_H,
            )

            # Separator between the two live panels.
            cv2.line(
                camera,
                (PANEL_W - 1, 0),
                (PANEL_W - 1, PANEL_H),
                (38, 46, 58),
                2,
            )
            cv2.line(
                mesh,
                (0, 0),
                (0, PANEL_H),
                (38, 46, 58),
                2,
            )

            output = np.hstack([camera, mesh])

        except Exception:
            # Keep the browser stream alive if an individual frame fails.
            output = np.hstack([
                camera,
                self._technical_background(PANEL_W, PANEL_H),
            ])

        return av.VideoFrame.from_ndarray(output, format="bgr24")

    def __del__(self):
        try:
            self.face_landmarker.close()
        except Exception:
            pass
        try:
            self.hand_landmarker.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
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
              Real-time facial landmark visualization with gesture-responsive
              rendering. The camera feed and technical mesh field run together
              in a single live vision pipeline.
            </p>
          </div>
          <div class="live-pill">
            <span class="live-dot"></span>LIVE VISION PIPELINE
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        rtc_configuration = build_rtc_configuration()
    except Exception as exc:
        st.error("Could not initialize the WebRTC configuration.")
        st.caption(
            "Check your Metered TURN credentials in Streamlit Secrets."
        )
        st.stop()

    st.markdown(
        '<div class="section-label"><span>LIVE ANALYSIS</span>'
        '<span>Camera + Mesh · 16:9</span></div>',
        unsafe_allow_html=True,
    )

    ctx = webrtc_streamer(
        key="face-mesh-studio-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=FaceMeshProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30, "max": 30},
            },
            "audio": False,
        },
        async_processing=True,
    )

    # Streamlit reruns can occur while the WebRTC component remains active.
    # These metrics are therefore intentionally lightweight.
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    running = bool(ctx.state.playing)

    with m1:
        st.markdown(
            f'<div class="metric"><div class="metric-k">Pipeline</div>'
            f'<div class="metric-v">{"LIVE" if running else "READY"}</div>'
            f'<div class="metric-s">WebRTC camera stream</div></div>',
            unsafe_allow_html=True,
        )

    with m2:
        gesture = (
            ctx.video_processor.gesture
            if ctx.video_processor is not None
            else "UNKNOWN"
        )
        st.markdown(
            f'<div class="metric"><div class="metric-k">Gesture</div>'
            f'<div class="metric-v">{gesture.replace("_", " ").title()}</div>'
            f'<div class="metric-s">Hand classification</div></div>',
            unsafe_allow_html=True,
        )

    with m3:
        face = (
            ctx.video_processor.face_detected
            if ctx.video_processor is not None
            else False
        )
        st.markdown(
            f'<div class="metric"><div class="metric-k">Face</div>'
            f'<div class="metric-v">{"LOCKED" if face else "SEARCHING"}</div>'
            f'<div class="metric-s">Single-face landmark model</div></div>',
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            '<div class="metric"><div class="metric-k">Model</div>'
            '<div class="metric-v">MediaPipe</div>'
            '<div class="metric-s">Tasks API · Face + Hand</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="margin-top:14px;border-top:1px solid #171e27;
        padding-top:14px;color:#566274;font-size:10px;line-height:1.6;">
        <b style="color:#7d899b;">INTERACTION</b>&nbsp;&nbsp;
        Fist → red &nbsp;·&nbsp; Open palm → green &nbsp;·&nbsp;
        Peace → blue &nbsp;·&nbsp; Thumbs up → yellow &nbsp;·&nbsp;
        Point → magenta
        </div>
        <div style="text-align:center;color:#424c5a;font-size:9px;
        letter-spacing:.12em;text-transform:uppercase;padding-top:20px;">
        FACE MESH STUDIO · REAL-TIME COMPUTER VISION
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
