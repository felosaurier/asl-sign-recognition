"""
Preprocessing pipeline for ASL dataset.

Filename convention: P{person}_{sign}_{index}.jpg
  e.g.  P1_1_1.jpg  ... P1_1_100.jpg
        P2_1_101.jpg ... P2_1_200.jpg
        ...
        P10_1_901.jpg ... P10_1_1000.jpg

Person-based split (no data leakage between persons):
  Train : P1 – P8  (800 images / class)
  Val   : P9       (100 images / class)
  Test  : P10      (100 images / class)

Two modes:
  --mode landmarks  -> extracts 21 MediaPipe hand landmarks  (recommended)
  --mode cnn        -> resizes images to IMG_SIZE x IMG_SIZE
"""

import re
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

try:
    from mediapipe_utils import create_static_detector, detect_static, landmarks_to_features
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE    = 64
DATASET_DIR = Path(__file__).parent.parent / "data" / "asl_dataset"
OUTPUT_DIR  = Path(__file__).parent.parent / "data"

# P1-P8 train | P9 val | P10 test
TRAIN_PERSONS = set(range(1, 9))
VAL_PERSONS   = {9}
TEST_PERSONS  = {10}

# Supported static labels (J and Z require motion → excluded)
STATIC_LABELS = set(list("ABCDEFGHIKLMNOPQRSTUVWXY") + [str(d) for d in range(10)])

# Regex to extract person number from filename, e.g. "P3_A_42.jpg" -> 3
_PERSON_RE = re.compile(r"^[Pp](\d+)_", )


def parse_person(filename: str) -> int | None:
    """Return the person index from a filename like P3_A_42.jpg, or None."""
    m = _PERSON_RE.match(filename)
    return int(m.group(1)) if m else None


# ── Label map ─────────────────────────────────────────────────────────────────

def build_label_map(dataset_dir: Path) -> dict[str, int]:
    folders = sorted(
        f.name.upper()
        for f in dataset_dir.iterdir()
        if f.is_dir() and f.name.upper() in STATIC_LABELS
    )
    return {label: idx for idx, label in enumerate(folders)}


# ── Image loader ──────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── CNN feature extraction ────────────────────────────────────────────────────

def preprocess_cnn(img: np.ndarray) -> np.ndarray:
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ── Landmark feature extraction ───────────────────────────────────────────────

def extract_landmarks(img_rgb: np.ndarray, detector) -> np.ndarray | None:
    """Returns (63,) float32 or None if no hand detected."""
    hand_landmarks = detect_static(detector, img_rgb)
    if hand_landmarks is None:
        return None
    return landmarks_to_features(hand_landmarks[0])


# ── Per-class dataset builder ─────────────────────────────────────────────────

def collect_samples(folder: Path, label_idx: int, mode: str, hands=None):
    """
    Returns three lists (train, val, test), each containing (features, label) tuples.
    Routing is done by person ID parsed from the filename.
    """
    train, val, test = [], [], []
    skipped = 0

    images = sorted(
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.png"))
        + list(folder.glob("*.jpeg"))
    )

    for img_path in images:
        person = parse_person(img_path.name)
        if person is None:
            # Fallback: no person tag → add to train
            person = 1

        img = load_image(img_path)
        if img is None:
            continue

        if mode == "cnn":
            features = preprocess_cnn(img)
        else:
            features = extract_landmarks(img, hands)
            if features is None:
                skipped += 1
                continue

        entry = (features, label_idx)
        if person in TRAIN_PERSONS:
            train.append(entry)
        elif person in VAL_PERSONS:
            val.append(entry)
        elif person in TEST_PERSONS:
            test.append(entry)
        # persons outside 1-10 → silently skip

    return train, val, test, skipped


# ── Full dataset builder ──────────────────────────────────────────────────────

def build_dataset(dataset_dir: Path, label_map: dict[str, int], mode: str):
    hands = None
    if mode == "landmarks":
        if not HAS_MEDIAPIPE:
            raise RuntimeError("mediapipe not installed — run: pip install mediapipe")
        hands = create_static_detector(min_detection_confidence=0.5)

    all_train, all_val, all_test = [], [], []
    total_skipped = 0

    for label, idx in label_map.items():
        folder = dataset_dir / label
        if not folder.exists():
            folder = dataset_dir / label.lower()
        if not folder.exists():
            print(f"  [WARN] Folder not found: {label}")
            continue

        tr, va, te, sk = collect_samples(folder, idx, mode, hands)
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)
        total_skipped += sk
        print(f"  {label:>2}  train={len(tr):>4}  val={len(va):>4}  test={len(te):>4}"
              + (f"  skipped={sk}" if sk else ""))

    if hands:
        hands.close()  # HandLandmarker also supports close()

    if total_skipped:
        print(f"\n  Total skipped (no hand detected): {total_skipped}")

    def unpack(pairs):
        if not pairs:
            return np.array([]), np.array([])
        X = np.array([p[0] for p in pairs], dtype=np.float32)
        y = np.array([p[1] for p in pairs], dtype=np.int32)
        return X, y

    return unpack(all_train), unpack(all_val), unpack(all_test)


# ── Save ──────────────────────────────────────────────────────────────────────

def save(train, val, test, label_map: dict[str, int], out_path: Path):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test

    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        label_map=np.array(list(label_map.keys())),
    )
    print(f"\nSaved -> {out_path}")
    print(f"  Train  : {len(X_train):>5} samples  (P1–P8)")
    print(f"  Val    : {len(X_val):>5} samples  (P9)")
    print(f"  Test   : {len(X_test):>5} samples  (P10)")
    print(f"  Classes: {list(label_map.keys())}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASL dataset preprocessor")
    parser.add_argument("--mode", choices=["cnn", "landmarks"], default="landmarks")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    label_map = build_label_map(dataset_dir)
    print(f"Found {len(label_map)} classes: {list(label_map.keys())}")
    print(f"Split: P1-P8 → train | P9 → val | P10 → test\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"dataset_{args.mode}.npz"

    print(f"[{args.mode.upper()} mode]")
    train, val, test = build_dataset(dataset_dir, label_map, args.mode)

    if len(train[0]) == 0:
        raise RuntimeError("No training samples collected. Check dataset path.")

    save(train, val, test, label_map, out_path)


if __name__ == "__main__":
    main()
