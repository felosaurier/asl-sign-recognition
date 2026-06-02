"""
Evaluation script — confusion matrix, per-class accuracy, misclassification analysis.

Usage:
    python src/evaluate.py --mode landmarks
    python src/evaluate.py --mode cnn
"""

import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR   = Path(__file__).parent.parent / "data"
FIG_DIR    = Path(__file__).parent.parent / "figures"


def evaluate(mode: str):
    # Load dataset
    npz_path = DATA_DIR / f"dataset_{mode}.npz"
    data = np.load(str(npz_path), allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    label_names    = data["label_map"].tolist()

    # Load model
    model_file = "asl_landmarks_best.keras" if mode == "landmarks" else "asl_cnn_best.keras"
    model = tf.keras.models.load_model(str(MODELS_DIR / model_file))

    # Predict
    probs  = model.predict(X_test, batch_size=128, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # Report
    print("\n── Classification Report ───────────────────────")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True",      fontsize=13)
    ax.set_title(f"Confusion Matrix — {mode.upper()} model", fontsize=15)
    plt.tight_layout()

    out_path = FIG_DIR / f"confusion_matrix_{mode}.png"
    fig.savefig(str(out_path), dpi=150)
    print(f"\nConfusion matrix saved -> {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["landmarks", "cnn"], default="landmarks")
    args = parser.parse_args()
    evaluate(args.mode)


if __name__ == "__main__":
    main()
