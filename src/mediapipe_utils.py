"""
MediaPipe Tasks API helper (compatible with mediapipe >= 0.10.13).
Handles model download and provides a unified interface for both
static images (preprocessing) and live video (inference).
"""

import urllib.request
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"


def ensure_model() -> Path:
    """Download the hand landmarker model if not already present."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model -> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")
    return MODEL_PATH


def create_static_detector(min_detection_confidence: float = 0.5):
    """Detector for still images (preprocessing)."""
    model_path = ensure_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def create_video_detector(min_detection_confidence: float = 0.6,
                          min_tracking_confidence: float = 0.5):
    """Detector for live webcam frames (inference)."""
    model_path = ensure_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def landmarks_to_features(hand_landmarks) -> np.ndarray:
    """
    Converts a list of NormalizedLandmark to a normalised (63,) float32 array.
    Identical normalisation in preprocess.py AND inference_realtime.py — must not diverge.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32,
    )
    coords -= coords[0]          # wrist as origin
    scale = np.abs(coords).max()
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()      # (63,)


def detect_static(detector, img_rgb: np.ndarray):
    """Run detection on a single RGB image array. Returns hand_landmarks list or None."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = detector.detect(mp_image)
    return result.hand_landmarks if result.hand_landmarks else None


def detect_video_frame(detector, img_rgb: np.ndarray, timestamp_ms: int):
    """Run detection on a video frame. Returns hand_landmarks list or None."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = detector.detect_for_video(mp_image, timestamp_ms)
    return result.hand_landmarks if result.hand_landmarks else None
