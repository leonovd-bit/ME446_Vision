"""
train.py — Augment dataset images and train a CNN shape classifier.

Usage:
    python train.py

Reads images from:
    dataset/circle/
    dataset/square/

Saves:
    shape_classifier.keras   — trained model
    training_plot.png        — accuracy / loss curves
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR   = "dataset"
CLASSES       = ["circle", "square"]
IMG_SIZE      = 128             # resize all crops to IMG_SIZE x IMG_SIZE
BATCH_SIZE    = 32
EPOCHS        = 30
VALIDATION_SPLIT = 0.2          # 20 % of data used for validation
MODEL_PATH    = "shape_classifier.keras"
PLOT_PATH     = "training_plot.png"
SEED          = 42
# ---------------------------------------------------------------------------


def load_dataset():
    """Load all .jpg images from DATASET_DIR/class folders.

    Returns:
        images : float32 array, shape (N, IMG_SIZE, IMG_SIZE, 3), values in [0,1]
        labels : int array, shape (N,)  — index into CLASSES list
    """
    import cv2

    images, labels = [], []
    for label_idx, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, cls)
        files  = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        if not files:
            print(f"  WARNING: no images found in {folder}")
            continue
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label_idx)
        print(f"  Loaded {len(files):4d} images  →  {cls}")

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def build_model(num_classes):
    """Build a small CNN with built-in augmentation layers."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="augmentation")

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss    = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss    = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=metrics,
    )
    return model, loss


def plot_history(history, path):
    """Save accuracy and loss curves to *path*."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path)
    print(f"  Training plot saved → {path}")


def main():
    import tensorflow as tf
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("=== Shape Classifier Training ===")
    print(f"Classes : {CLASSES}")
    print(f"IMG_SIZE: {IMG_SIZE}x{IMG_SIZE}  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print()

    # --- Load data ---
    print("Loading dataset...")
    images, labels = load_dataset()
    print(f"  Total images: {len(images)}\n")

    if len(images) == 0:
        print("ERROR: No images found. Run collect_data.py first.")
        return

    # --- Shuffle ---
    idx = np.random.permutation(len(images))
    images, labels = images[idx], labels[idx]

    # --- Train / val split ---
    split = int(len(images) * (1 - VALIDATION_SPLIT))
    x_train, x_val = images[:split], images[split:]
    y_train, y_val = labels[:split], labels[split:]
    print(f"Train samples : {len(x_train)}")
    print(f"Val   samples : {len(x_val)}\n")

    # --- Build model ---
    print("Building model...")
    model, _ = build_model(num_classes=len(CLASSES))
    model.summary()
    print()

    # --- Callbacks ---
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    # --- Train ---
    print("Training...\n")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # --- Save ---
    model.save(MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")

    # --- Plot ---
    plot_history(history, PLOT_PATH)

    # --- Final metrics ---
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"\nFinal val accuracy : {val_acc * 100:.1f} %")
    print(f"Final val loss     : {val_loss:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
