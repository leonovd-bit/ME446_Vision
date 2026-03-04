"""
augment_and_train.py — Data augmentation and CNN training for shape classification.

Trains a binary classifier (circle vs. square) using images collected by
collect_data.py.  The best model is saved to best_model.keras.

Usage:
    python augment_and_train.py

Requirements:
    pip install tensorflow opencv-python
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
MODEL_PATH = "best_model.keras"
# ---------------------------------------------------------------------------


def build_generators():
    """Create training and validation ImageDataGenerators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.1,
        brightness_range=(0.85, 1.15),
        fill_mode="constant",
        cval=255,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_gen, val_gen


def build_model():
    """Build and compile the CNN."""
    model = keras.Sequential(
        [
            # Input
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

            # Block 1 — 32 filters
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 2 — 64 filters
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 3 — 128 filters
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Classifier head
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=== Shape Classification Training ===")
    print(f"Dataset : {DATASET_DIR}")
    print(f"Image size: {IMG_SIZE}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}\n")

    train_gen, val_gen = build_generators()

    print(f"Class indices: {train_gen.class_indices}")
    print(f"Training samples  : {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}\n")

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
