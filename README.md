# Face Mesh + Hand Gesture Color Control

Split-screen live app: left panel shows your webcam feed with hand
landmarks; right panel shows the live face mesh wireframe, recolored
based on the hand gesture you make.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

Press `q` in the window to quit.

## Gestures -> Colors

| Gesture     | How to make it                          | Mesh color |
|-------------|------------------------------------------|------------|
| Fist        | All fingers curled                       | Red        |
| Open palm   | All fingers extended                     | Green      |
| Peace       | Index + middle up, others down           | Blue       |
| Thumbs up   | Only thumb extended                      | Yellow     |
| Point       | Only index finger extended               | Magenta    |
| (anything else) | -                                    | Gray (default) |

## Known reliability limits (read before demoing)

This uses a **rule-based** classifier on 2D hand landmarks (finger
tip vs. joint y-position, thumb via x-position). It is not a trained
gesture-recognition model, so:

- **Thumb detection is the weakest point.** It assumes a roughly
  frontal hand orientation. Rotate your hand 30-40 degrees and it
  will misclassify thumb state.
- **Peace vs. Point can be confused** at the frame where a finger is
  transitioning — the 8-frame majority-vote smoother reduces flicker
  but does not eliminate misreads during the transition itself.
- **Lighting and occlusion** (fingers overlapping each other, hand
  partially out of frame) degrade MediaPipe's underlying landmark
  detection before your gesture logic even runs.
- If you need higher reliability for a demo, the next step up is
  collecting labeled gesture samples and training a small classifier
  (e.g. an MLP on the 21 landmark coordinates) instead of hand-coded
  rules — happy to help build that if this matters for the final
  version.

## Important: mediapipe API change (Aug 2026)

Current mediapipe releases (0.10.30+, and the 1.x line) have **removed
the legacy `mediapipe.solutions` API entirely** (`mp.solutions.face_mesh`,
`mp.solutions.hands`, `mp.solutions.drawing_utils` — none of it exists
anymore; confirmed directly against the installed package, not from
memory). This project now uses the newer Tasks API
(`FaceLandmarker` / `HandLandmarker`) instead, in `landmarkers.py`.

Practical implications:

- **First run downloads two small model files** (`face_landmarker.task`,
  `hand_landmarker.task`) from Google's model hosting and caches them in
  the system temp dir. This needs outbound internet access once. I
  verified the download URLs against Google's own docs and the official
  `mediapipe-samples` GitHub repo, but couldn't test the actual download
  in the environment I built this in (no access to
  `storage.googleapis.com` there) — the first real run is the real test
  of that step.
- The face-mesh wireframe is now a **live Delaunay triangulation**
  (via `cv2.Subdiv2D`) of the detected landmarks, computed fresh each
  frame — not the old fixed `FACEMESH_TESSELATION` edge list, since that
  data isn't shipped anywhere in current mediapipe at all.
- Hand landmarks are drawn manually (`mp.solutions.drawing_utils` is
  also gone) — simple green dots per joint, no bone-connection lines.
- **Do not pin an older mediapipe version to get `solutions` back** —
  I tested this: mediapipe 0.10.14 still has `mp.solutions.face_mesh`,
  but (a) it hard-depends on `opencv-contrib-python`, which reintroduces
  the `libGL.so.1` error you just fixed, and (b) its protobuf
  requirement conflicts with Streamlit's. Both are real, verified
  problems, not theoretical.

## Streamlit version (browser, deploy-ready)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This uses `streamlit-webrtc`, so video runs as real browser-side WebRTC
streaming rather than a polling loop — it's the version to deploy
(e.g. Streamlit Community Cloud), not `main.py` (which uses
`cv2.imshow()` and only works as a local desktop window).

**Before you deploy this anywhere other than localhost:**

- WebRTC needs to establish a peer connection between the visitor's
  browser and the server. This app is configured with Google's public
  STUN server, which is enough on most home/office networks. On
  restrictive networks (corporate firewalls, some mobile carriers,
  most cloud NAT setups) STUN alone isn't enough and connections will
  fail silently — you'd need a TURN server (e.g. via Twilio, or your
  own coturn instance) added to `RTC_CONFIGURATION` in
  `streamlit_app.py`.
- I verified `streamlit`, `streamlit-webrtc`, `av`, and `aiortc`
  install and import cleanly, and re-ran the gesture-classification
  unit test against the shared logic module — but I could not test
  the actual video pipeline end-to-end (no camera in the environment
  I built this in). The first time you run it locally is the real test
  of the WebRTC + MediaPipe integration.
- Panel size is smaller here (480x360 vs 640x480 in the desktop
  version) to keep the WebRTC-streamed frame size reasonable — combined
  frame is 960x360. Adjust `PANEL_W`/`PANEL_H` in `streamlit_app.py`
  if you want it bigger, at the cost of more bandwidth per frame.

## Deploying to Streamlit Community Cloud: libGL error

If you see `ImportError: libGL.so.1: cannot open shared object file`,
it's because the full `opencv-python` package links against GUI/OpenGL
libraries the server container doesn't have (and doesn't need, since
there's no display). This repo already uses the fix:

- `requirements.txt` uses `opencv-python-headless` instead of
  `opencv-python` (same API, no GL dependency).
- `packages.txt` lists `libgl1` — Streamlit Cloud reads this file and
  runs `apt-get install` on its contents before starting your app,
  covering the lower-level GL library mediapipe still needs even with
  the headless opencv build.
  (An earlier version of this file also listed `libglib2.0-0`, which
  broke the apt install on Streamlit Cloud's current base image due to
  an unrelated dependency conflict with `libffi7`. Removed — `libgl1`
  alone resolves the `libGL.so.1` error.)

If you added `opencv-python` back in yourself at some point, remove it
— having both installed can leave a broken `cv2` module behind.

## Files

- `main.py` — desktop OpenCV window version
- `streamlit_app.py` — Streamlit + streamlit-webrtc browser version
- `gesture_logic.py` — shared, unit-tested gesture classification and
  rendering logic used by both apps
- `requirements.txt` — dependencies for both versions
- `packages.txt` — system (apt) packages required on Streamlit Cloud
