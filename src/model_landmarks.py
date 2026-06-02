"""
MLP model for ASL classification from MediaPipe hand landmarks.

Input:  (batch, 63)  — 21 landmarks × (x, y, z), pre-normalised
Output: (batch, num_classes)  — softmax probabilities

Why an MLP works well here:
  - Landmarks are already a structured, low-dimensional feature vector.
  - Training is 100× faster than CNN.
  - Invariant to background, lighting, and skin colour.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_landmark_mlp(
    num_classes: int,
    input_dim: int = 63,
    hidden_units: tuple[int, ...] = (512, 256, 128),
    dropout_rate: float = 0.5,   # increased from 0.3 → stronger regularisation
) -> keras.Model:
    """
    Deep MLP with:
      - Batch normalisation for stable training
      - Residual connections between same-width layers
      - L2 + Dropout for regularisation
    """
    inputs = keras.Input(shape=(input_dim,), name="landmark_input")

    # Small Gaussian noise during training simulates different hand sizes/styles
    x = layers.GaussianNoise(0.02)(inputs)
    x = layers.BatchNormalization()(x)

    prev = None
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=None,
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"dense_{i}",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout_rate)(x)

        # Residual connection when width is unchanged
        if prev is not None and prev.shape[-1] == units:
            x = layers.Add()([x, prev])

        prev = x

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ASL_LandmarkMLP")
    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        # label_smoothing=0.1 prevents overconfident predictions → better generalisation
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_landmark_mlp(num_classes=34)
    compile_model(model)
    model.summary()
