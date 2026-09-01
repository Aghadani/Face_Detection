# Face Mesh + Hand Gesture Color Control

Detects your face and hand, draws a live face-mesh wireframe next to
your camera feed, and recolors the mesh based on your hand gesture.

Two versions in this repo:

- `streamlit_app.py` — browser app, continuous live video via
  `streamlit-webrtc`, using a free TURN relay (see below).
- `main.py` — desktop OpenCV window version, continuous live video via
  your local webcam. Run with `python main.py`, not through Streamlit.

## Live video + free TURN relay

Streamlit Community Cloud's network does not complete a WebRTC
connection with STUN alone (confirmed against `streamlit-webrtc`'s own
maintainer's documented findings) — it needs a TURN relay. This app
uses a free TURN relay, no paid service, no VPS to run yourself.

**Default: zero-signup, works immediately, but less reliable.**
Uses Open Relay Project's static public TURN credentials
(`openrelayproject`/`openrelayproject`). Genuinely free, no account
needed — but it's a shared, unauthenticated public resource, confirmed
via a real user report that it can silently stop working under load,
and `streamlit-webrtc`'s own maintainer flags it as unreliable. If
your video won't connect, this relay being overloaded is the most
likely reason, not a bug in the app.

**More reliable, still free: a Metered account.** The static relay
above just failed again in your log with the same connection-teardown
error as before — that confirms it, this isn't reliable enough to
depend on. Steps to switch to the reliable free path:

1. Sign up free at https://dashboard.metered.ca/signup (email-based,
   not phone verification).
2. Note your app name from the dashboard sidebar (your domain will be
   `<that-name>.metered.live`).
3. Go to the **TURN Server** page in the dashboard and click
   **"Generate Your First Credential"** (or **"Add Credential"**).
4. Once created, click **"Show API Key"** next to that credential —
   this is a credential-scoped key, safe to use directly (not your
   account's Secret Key).
5. Set these as Streamlit secrets (Settings -> Secrets on Streamlit
   Cloud, or `.streamlit/secrets.toml` locally):

```toml
METERED_APP_NAME = "your-app-name"   # from step 2
METERED_API_KEY = "your-api-key"     # from step 4
```

`ice_servers.py` fetches these credentials automatically when the
secrets are set, and falls back to the static public relay if they're
missing or the request fails — so the app degrades gracefully rather
than breaking if you remove the secrets later.

I verified the fallback TURN hostname resolves via DNS, but a full
TURN relay handshake needs live UDP traffic I can't test from the
environment I build this in — your actual deployment is the real test
of whether either path connects reliably for you.

If neither free option is reliable enough, the remaining paths are a
self-hosted coturn server (needs your own VPS — ask if you want this
built out) or a paid TURN service.

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

Gesture classification is smoothed over the last 8 frames (majority
vote) to reduce flicker between individual misreads.

## Known reliability limits

This uses a **rule-based** classifier on 2D hand landmarks, not a
trained gesture-recognition model:

- **Thumb detection is the weakest point** — assumes a roughly frontal
  hand orientation; rotate your hand 30-40 degrees and it misclassifies.
  `OPEN_PALM` and `FIST` no longer require thumb agreement (an earlier
  version did, and a clear open-palm or fist would misclassify as
  `UNKNOWN` whenever the thumb read ambiguously — very common even in
  an unambiguous real pose). The remaining trade-off: `THUMBS_UP` is
  checked first since it's the one gesture defined by thumb state alone,
  so a real fist where the thumb *falsely* reads as "up" can register
  as `THUMBS_UP` instead of `FIST`.
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

If you see `ImportError`/`OSError` for `libGL.so.1`, `libEGL.so.1`,
`libGLESv2.so.2`, or `libgthread-2.0.so.0`: mediapipe's compiled C
bindings link against the full GL/EGL/GLES stack even for CPU-only
inference, and `cv2` itself needs glib's threading support.
`requirements.txt` uses `opencv-python-headless` (no GUI/GL dependency
itself), and `packages.txt` lists the system libraries still needed:
`libgl1`, `libegl1`, `libgles2`, `libglib2.0-0t64`.

Note the `t64` suffix on the glib package — that's not a typo. Debian
trixie (the base image Streamlit Cloud uses) renamed `libglib2.0-0` to
`libglib2.0-0t64` as part of a distro-wide 64-bit `time_t` ABI
transition (confirmed via Debian's own package pages and bug tracker).
The old name (`libglib2.0-0`) still exists as an orphaned package on
this image with a broken dependency on `libffi7` that isn't
installable — which is exactly the apt error you'd get if you tried
adding it back under the old name.

If a related library in this same family shows up missing next, it's
the same pattern — add it to `packages.txt` the same way.

## Desktop version (`main.py`)

```bash
python main.py
```

True continuous live video via `cv2.imshow()` — a local desktop window,
not deployable to a web server. Press `q` to quit.

## Files

- `streamlit_app.py` — Streamlit browser app (continuous live video via
  WebRTC + free TURN relay)
- `ice_servers.py` — TURN/STUN config: Metered API key if configured,
  falls back to free static public credentials
- `main.py` — desktop OpenCV window version (continuous live video,
  local only)
- `gesture_logic.py` — shared, unit-tested gesture classification and
  mesh-rendering logic used by both apps
- `landmarkers.py` — Tasks API model loading/creation, shared by both
  apps
- `requirements.txt` — Python dependencies
- `packages.txt` — system (apt) packages required on Streamlit Cloud
