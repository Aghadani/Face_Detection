"""
Landmarker setup using MediaPipe's Tasks API.

The legacy `mediapipe.solutions.face_mesh` / `mediapipe.solutions.hands`
API used in earlier versions of this project no longer exists in current
mediapipe releases (confirmed: mediapipe>=0.10.30 and 1.x ship only
`mediapipe.tasks`, with no `mediapipe.solutions` module at all). This
module builds FaceLandmarker / HandLandmarker from the Tasks API instead,
which is the actively maintained replacement.

Both landmarkers require a `.task` model bundle file. These are not
bundled with the pip package -- they're downloaded on first run and
cached locally. URLs below are Google's official model hosting,
confirmed via https://developers.google.com/edge/mediapipe/solutions/vision
and the official mediapipe-samples repo (as of Aug 2026).
"""

import os
import tempfile
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Use the system temp dir rather than a path inside the repo checkout --
# some deployment platforms (Streamlit Cloud included) may not guarantee
# the repo directory is writable at runtime.
MODEL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "mp_gesture_models")


def _ensure_model(url, filename):
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    dest = os.path.join(MODEL_CACHE_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp_dest = dest + ".part"
    urllib.request.urlretrieve(url, tmp_dest)
    os.replace(tmp_dest, dest)  # atomic-ish rename once fully downloaded
    return dest


def create_face_landmarker(running_mode=mp_vision.RunningMode.VIDEO):
    """Returns a FaceLandmarker. running_mode=VIDEO for a continuous frame
    stream with increasing timestamps; running_mode=IMAGE for independent,
    unconnected single photos (e.g. st.camera_input snapshots) where there
    is no temporal continuity between captures to track."""
    model_path = _ensure_model(FACE_MODEL_URL, "face_landmarker.task")
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def create_hand_landmarker(running_mode=mp_vision.RunningMode.VIDEO):
    """Returns a HandLandmarker. See create_face_landmarker() for the
    running_mode explanation."""
    model_path = _ensure_model(HAND_MODEL_URL, "hand_landmarker.task")
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_hands=1,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def to_image(rgb_frame):
    """Wrap an RGB numpy array as an mp.Image for detect_for_video()."""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
