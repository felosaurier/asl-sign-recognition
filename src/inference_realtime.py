"""
Real-time ASL recognition via webcam.

Recommended pipeline (default):
  MediaPipe hand landmarks  ->  MLP model  ->  letter + confidence overlay

Fallback:
  ROI crop  ->  CNN model  ->  letter + confidence overlay

Usage:
    python src/inference_realtime.py                   # landmark mode (default)
    python src/inference_realtime.py --mode cnn        # CNN mode
    python src/inference_realtime.py --cam 1           # second webcam
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
MODELS_DIR     = Path(__file__).parent.parent / "models"
IMG_SIZE       = 64          # must match preprocess.py
CONF_THRESHOLD = 0.70        # minimum confidence to display prediction
SMOOTHING_N    = 7           # vote over last N frames to reduce flicker
FONT           = cv2.FONT_HERSHEY_DUPLEX

# Colours (BGR)
CLR_GREEN  = (0, 220, 80)
CLR_RED    = (0, 60, 220)
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0, 0, 0)
CLR_YELLOW = (0, 220, 220)
CLR_HAND   = (80, 220, 255)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model_and_labels(mode: str):
    import tensorflow as tf

    label_map_path = MODELS_DIR / "label_map.npy"
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map.npy not found in {MODELS_DIR}")
    label_names = np.load(str(label_map_path), allow_pickle=True).tolist()

    model_file = "asl_landmarks_best.keras" if mode == "landmarks" else "asl_cnn_best.keras"
    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\nRun: python src/train.py --mode {mode}"
        )

    model = tf.keras.models.load_model(str(model_path))
    print(f"Loaded model: {model_path}")
    print(f"Classes: {label_names}")
    return model, label_names


# ── MediaPipe setup ────────────────────────────────────────────────────────────

def init_mediapipe():
    from mediapipe_utils import create_video_detector, landmarks_to_features as lm_to_feat
    detector = create_video_detector(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return detector


# ── Feature extraction ────────────────────────────────────────────────────────

def landmarks_to_features(hand_landmarks) -> np.ndarray:
    """Thin wrapper — delegates to mediapipe_utils for identical normalisation."""
    from mediapipe_utils import landmarks_to_features as _lm
    return _lm(hand_landmarks)


def crop_hand_roi(frame_rgb: np.ndarray, hand_landmarks, padding: float = 0.20) -> np.ndarray | None:
    """Crop a square ROI around the detected hand for CNN mode."""
    h, w = frame_rgb.shape[:2]
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    x_min = max(0, int((min(xs) - padding) * w))
    x_max = min(w, int((max(xs) + padding) * w))
    y_min = max(0, int((min(ys) - padding) * h))
    y_max = min(h, int((max(ys) + padding) * h))

    roi = frame_rgb[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None
    return cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0


# ── Prediction with temporal smoothing ───────────────────────────────────────

class SmoothPredictor:
    def __init__(self, n: int, num_classes: int):
        self._votes    = deque(maxlen=n)
        self._n        = n
        self._nc       = num_classes

    def update(self, probs: np.ndarray) -> tuple[int, float]:
        self._votes.append(probs)
        avg = np.mean(self._votes, axis=0)
        idx = int(np.argmax(avg))
        return idx, float(avg[idx])

    def reset(self):
        self._votes.clear()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1):
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)


def draw_overlay(frame, letter: str, conf: float, fps: float, hand_detected: bool):
    h, w = frame.shape[:2]

    # ── FPS badge (top-right) ─────────────────────────────────────────────────
    fps_text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(fps_text, FONT, 0.55, 1)
    cv2.rectangle(frame, (w - tw - 16, 8), (w - 4, th + 16), CLR_BLACK, -1)
    cv2.putText(frame, fps_text, (w - tw - 10, th + 10), FONT, 0.55, CLR_YELLOW, 1)

    if not hand_detected:
        msg = "No hand detected"
        (mw, mh), _ = cv2.getTextSize(msg, FONT, 0.8, 2)
        cv2.putText(frame, msg, ((w - mw) // 2, h // 2), FONT, 0.8, CLR_RED, 2)
        return

    # ── Prediction card (bottom centre) ───────────────────────────────────────
    conf_pct  = int(conf * 100)
    bar_color = CLR_GREEN if conf >= CONF_THRESHOLD else CLR_RED
    display   = letter if conf >= CONF_THRESHOLD else "?"

    # Background card
    card_w, card_h = 200, 100
    cx = (w - card_w) // 2
    cy = h - card_h - 20
    overlay = frame.copy()
    draw_rounded_rect(overlay, cx, cy, cx + card_w, cy + card_h, 12, CLR_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Letter
    (lw, lh), _ = cv2.getTextSize(display, FONT, 3.2, 4)
    cv2.putText(frame, display,
                (cx + (card_w - lw) // 2, cy + lh + 10),
                FONT, 3.2, CLR_WHITE, 4)

    # Confidence bar
    bar_x, bar_y = cx + 10, cy + card_h - 22
    bar_total = card_w - 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_total, bar_y + 10), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_total * conf), bar_y + 10), bar_color, -1)
    cv2.putText(frame, f"{conf_pct}%", (bar_x + bar_total + 4, bar_y + 10),
                FONT, 0.45, CLR_WHITE, 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(mode: str = "landmarks", cam_index: int = 0):
    model, label_names = load_model_and_labels(mode)
    num_classes = len(label_names)
    predictor   = SmoothPredictor(SMOOTHING_N, num_classes)

    detector = init_mediapipe()
    from mediapipe_utils import detect_video_frame

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\nASL Recognition running — press Q to quit\n")

    fps_deque     = deque(maxlen=30)
    prev_time     = time.perf_counter()
    letter, conf  = "?", 0.0
    hand_detected = False
    frame_idx     = 0

    # Connections for manual skeleton drawing (Tasks API)
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — exiting.")
            break

        frame     = cv2.flip(frame, 1)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = int(frame_idx * (1000 / 30))   # ms, assumes 30 fps
        frame_idx += 1

        hand_landmarks_list = detect_video_frame(detector, rgb, timestamp)

        if hand_landmarks_list:
            hand_lm = hand_landmarks_list[0]   # list of NormalizedLandmark
            hand_detected = True

            # Draw skeleton manually
            h, w = frame.shape[:2]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], CLR_HAND, 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, CLR_WHITE, -1)
                cv2.circle(frame, pt, 4, CLR_HAND,   1)

            # Extract features
            if mode == "landmarks":
                features = landmarks_to_features(hand_lm)[np.newaxis, :]
            else:
                roi = crop_hand_roi(rgb, hand_lm)
                if roi is None:
                    hand_detected = False
                    predictor.reset()
                    continue
                features = roi[np.newaxis, ...]

            probs     = model.predict(features, verbose=0)[0]
            idx, conf = predictor.update(probs)
            letter    = label_names[idx]
        else:
            hand_detected = False
            predictor.reset()
            letter, conf = "?", 0.0

        # FPS calculation
        now       = time.perf_counter()
        fps_deque.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps       = np.mean(fps_deque)

        draw_overlay(frame, letter, conf, fps, hand_detected)

        cv2.imshow("ASL Sign Recognition  |  Q = quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Session ended.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time ASL recognition")
    parser.add_argument("--mode", choices=["landmarks", "cnn"], default="landmarks")
    parser.add_argument("--cam",  type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    run(mode=args.mode, cam_index=args.cam)


if __name__ == "__main__":
    main()
