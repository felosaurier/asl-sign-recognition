"""
Training script — supports both CNN and Landmark modes.

Usage:
    python src/train.py --mode landmarks
    python src/train.py --mode cnn
"""

import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR   = Path(__file__).parent.parent / "data"

EPOCHS      = 80
BATCH_SIZE  = 64
LR_INITIAL  = 1e-3
PATIENCE    = 12         # early-stopping patience
MIN_LR      = 1e-6


def load_dataset(mode: str):
    npz_path = DATA_DIR / f"dataset_{mode}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {npz_path}.\n"
            f"Run: python src/preprocess.py --mode {mode}"
        )
    data = np.load(npz_path, allow_pickle=True)
    label_names = data["label_map"].tolist()
    return (
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        data["X_test"],  data["y_test"],
        label_names,
    )


def build_callbacks(model_path: Path):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",      # loss is more stable than accuracy
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",      # stop when val_loss stops improving
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=MIN_LR,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(MODELS_DIR / "logs"),
            histogram_freq=1,
        ),
    ]


def to_onehot(y, num_classes):
    return np.eye(num_classes, dtype=np.float32)[y]


def train_landmarks():
    from model_landmarks import build_landmark_mlp, compile_model

    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_dataset("landmarks")
    num_classes = len(label_names)
    print(f"Classes ({num_classes}): {label_names}")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # CategoricalCrossentropy with label_smoothing needs one-hot labels
    Y_train = to_onehot(y_train, num_classes)
    Y_val   = to_onehot(y_val,   num_classes)

    model = build_landmark_mlp(num_classes=num_classes)
    compile_model(model)
    model.summary()

    model_path = MODELS_DIR / "asl_landmarks_best.keras"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=build_callbacks(model_path),
    )

    print("\n── Test evaluation ──────────────────────────────")
    # evaluate with integer labels → use sparse_categorical_accuracy
    Y_test_oh = to_onehot(y_test, num_classes)
    loss, acc = model.evaluate(X_test, Y_test_oh, verbose=0)
    print(f"  Loss    : {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    # Save label map alongside model
    np.save(str(MODELS_DIR / "label_map.npy"), np.array(label_names))
    return history


def train_cnn():
    from model_cnn import build_cnn, compile_model

    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_dataset("cnn")
    num_classes = len(label_names)
    print(f"Classes ({num_classes}): {label_names}")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model = build_cnn(num_classes=num_classes)
    compile_model(model, num_classes=num_classes)
    model.summary()

    model_path = MODELS_DIR / "asl_cnn_best.keras"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=build_callbacks(model_path),
    )

    print("\n── Test evaluation ──────────────────────────────")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss    : {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    np.save(str(MODELS_DIR / "label_map.npy"), np.array(label_names))
    return history


def main():
    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument("--mode", choices=["cnn", "landmarks"], default="landmarks")
    args = parser.parse_args()

    print(f"TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available     : {len(gpus)}")

    if args.mode == "landmarks":
        train_landmarks()
    else:
        train_cnn()


if __name__ == "__main__":
    main()
