"""
CNN model for direct image classification of ASL hand signs.

Architecture: lightweight MobileNet-style depthwise-separable CNN.
Input:  (batch, 64, 64, 3)
Output: (batch, num_classes)  — softmax probabilities
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn(num_classes: int, img_size: int = 64, dropout_rate: float = 0.4) -> keras.Model:
    """
    Small but effective CNN with:
      - Depthwise separable convolutions (efficient for real-time)
      - Batch normalisation after every conv block
      - Global Average Pooling instead of Flatten (reduces parameter count)
      - L2 regularisation + Dropout to combat overfitting
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image_input")

    # ── Data augmentation (applied only during training) ──────────────────────
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.15)(x)
    x = layers.RandomContrast(0.15)(x)

    # ── Stem ──────────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # ── Depthwise-Separable blocks ────────────────────────────────────────────
    for filters in [64, 128, 128, 256]:
        x = _dw_sep_block(x, filters)

    # ── Classifier head ───────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ASL_CNN")
    return model


def _dw_sep_block(x, filters: int):
    """Depthwise separable conv block with residual connection if shapes match."""
    residual = x

    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 1, use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Project residual if channel count changed
    if residual.shape[-1] != filters:
        residual = layers.Conv2D(filters, 1, use_bias=False)(residual)
        residual = layers.BatchNormalization()(residual)

    return layers.Add()([x, residual])


def compile_model(model: keras.Model, num_classes: int, learning_rate: float = 1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_cnn(num_classes=34)
    compile_model(model, num_classes=34)
    model.summary()
