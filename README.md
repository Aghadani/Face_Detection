# Face Mesh + Hand Gesture Color Control

Detects your face and hand, draws a live face-mesh wireframe next to
your camera feed, and recolors the mesh based on your hand gesture.

Two versions in this repo:

- `streamlit_app.py` — browser app, **no external dependencies**.
  Snapshot-based: click "Take Photo," see the result, click again for
  the next frame. Not continuous live video (see below for why).
- `main.py` — desktop OpenCV window version, continuous live video via
  your local webcam. Run with `python main.py`, not through Streamlit.

## Why the Streamlit version is snapshot-based, not live video

Continuous live webcam video into a Python backend running on a server
(not your own machine) requires WebRTC. An earlier version of this app
used `streamlit-webrtc` for that. It failed to connect on Streamlit
Community Cloud — confirmed against `streamlit-webrtc`'s own
maintainer's documented findings: Streamlit Community Cloud's network
does not complete a WebRTC peer connection with a STUN server alone: it
requires a TURN relay, which means a third-party service (e.g. Twilio)
or a self-hosted TURN server.

Since no third-party service is being used, this version drops WebRTC
entirely and uses `st.camera_input` instead — a plain HTTP photo
upload, no peer connection, no STUN/TURN, works reliably on Streamlit
Cloud. The real trade-off: it's click-to-capture, not truly live.

If you later decide a third-party TURN service is acceptable, the
WebRTC + Twilio version can be rebuilt — ask and I'll put it back.

## Setup

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

First load downloads two small model files (`face_landmarker.task`,
`hand_landmarker.task`) from Google's model hosting and caches them in
the system temp dir — needs outbound internet access once.

## Gestures -> Colors

| Gesture     | How to make it                          | Mesh color |
|-------------|------------------------------------------|------------|
| Fist        | All fingers curled                       | Red        |
| Open palm   | All fingers extended                     | Green      |
| Peace       | Index + middle up, others down           | Blue       |
| Thumbs up   | Only thumb extended                      | Yellow     |
| Point       | Only index finger extended               | Magenta    |
| (anything else) | -                                    | Gray (default) |

Each snapshot is classified independently — there's no frame-to-frame
smoothing the way the desktop version has, since separate photos aren't
a continuous stream.

## Known reliability limits

This uses a **rule-based** classifier on 2D hand landmarks, not a
trained gesture-recognition model:

- **Thumb detection is the weakest point** — assumes a roughly frontal
  hand orientation; rotate your hand 30-40 degrees and it misclassifies.
- **Peace vs. Point can be confused** at ambiguous hand angles.
- **Lighting and occlusion** degrade MediaPipe's landmark detection
  before your gesture logic even runs.
- If you need higher reliability, the next step up is a trained
  classifier (e.g. an MLP on the 21 landmark coordinates) instead of
  hand-coded rules.

## Important: mediapipe API change (Aug 2026)

Current mediapipe releases (0.10.30+, and the 1.x line) have **removed
the legacy `mediapipe.solutions` API entirely** (`mp.solutions.face_mesh`,
`mp.solutions.hands`, `mp.solutions.drawing_utils` — none of it exists
anymore). This project uses the newer Tasks API (`FaceLandmarker` /
`HandLandmarker`) instead, in `landmarkers.py`.

- The face-mesh wireframe is a **live Delaunay triangulation**
  (`cv2.Subdiv2D`) of the detected landmarks, computed fresh each
  frame/photo — not the old fixed `FACEMESH_TESSELATION` edge list,
  since that data isn't shipped anywhere in current mediapipe.
- Hand landmarks are drawn manually (green dots per joint) since
  `mp.solutions.drawing_utils` is also gone.
- **Do not pin an older mediapipe version to get `solutions` back** —
  tested: 0.10.14 still has it, but hard-depends on
  `opencv-contrib-python` (reintroduces the `libGL.so.1` error) and
  conflicts with Streamlit's protobuf requirement.

## Deploying to Streamlit Community Cloud: native library errors

If you see `ImportError`/`OSError` for `libGL.so.1`, `libEGL.so.1`, or
`libGLESv2.so.2`: mediapipe's compiled C bindings link against the full
GL/EGL/GLES stack even for CPU-only inference. `requirements.txt` uses
`opencv-python-headless` (no GUI/GL dependency itself), and
`packages.txt` lists the system libraries mediapipe's native code still
needs: `libgl1`, `libegl1`, `libgles2`. If a related library in this
same family shows up missing, it's the same pattern — add it to
`packages.txt` the same way.

## Desktop version (`main.py`)

```bash
python main.py
```

True continuous live video via `cv2.imshow()` — a local desktop window,
not deployable to a web server. Press `q` to quit.

## Files

- `streamlit_app.py` — Streamlit browser app (snapshot-based, no
  external dependencies)
- `main.py` — desktop OpenCV window version (continuous live video,
  local only)
- `gesture_logic.py` — shared, unit-tested gesture classification and
  mesh-rendering logic used by both apps
- `landmarkers.py` — Tasks API model loading/creation, shared by both
  apps
- `requirements.txt` — Python dependencies
- `packages.txt` — system (apt) packages required on Streamlit Cloud
