"""
augment.py — Augment dataset images and split into train / test sets.

Usage:
    python augment.py

Reads images from:
    dataset/circle/
    dataset/square/

Outputs:
    augmented_dataset/
        circle/
            training/
            testing/
        square/
            training/
            testing/

Each original image produces several augmented variants.
75 % go to training, 25 % go to testing (split on originals before augmentation
so the same real photo never appears in both sets).
"""

import os
import random
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR   = "dataset"
OUTPUT_DIR    = "augmented_dataset"
CLASSES       = ["circle", "square"]
TRAIN_SPLIT   = 0.75            # fraction of originals going to training
AUGMENTS_PER_IMAGE = 5          # how many augmented copies to make per original
IMG_SIZE      = 128             # resize before augmenting
SEED          = 42
# ---------------------------------------------------------------------------


def ensure_dirs():
    for cls in CLASSES:
        for split in ("training", "testing"):
            os.makedirs(os.path.join(OUTPUT_DIR, cls, split), exist_ok=True)


def random_flip(img):
    """Randomly flip horizontally, vertically, or both."""
    mode = random.choice([-1, 0, 1])
    return cv2.flip(img, mode)


def random_rotate(img):
    """Rotate by a random angle in [-30, 30] degrees."""
    h, w = img.shape[:2]
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_zoom(img):
    """Zoom in by a random factor in [1.0, 1.3] then crop back to original size."""
    h, w = img.shape[:2]
    factor = random.uniform(1.0, 1.3)
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(img, (new_w, new_h))
    # Centre-crop back to original size
    y0 = (new_h - h) // 2
    x0 = (new_w - w) // 2
    return resized[y0:y0 + h, x0:x0 + w]


def random_brightness(img):
    """Shift brightness by a random amount in [-50, 50]."""
    delta = random.randint(-50, 50)
    out = img.astype(np.int16) + delta
    return np.clip(out, 0, 255).astype(np.uint8)


def random_contrast(img):
    """Scale contrast by a random factor in [0.7, 1.3]."""
    factor = random.uniform(0.7, 1.3)
    mean = img.mean()
    out = mean + factor * (img.astype(np.float32) - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(img):
    """Apply all augmentations in a random order."""
    ops = [random_flip, random_rotate, random_zoom,
           random_brightness, random_contrast]
    random.shuffle(ops)
    for op in ops:
        img = op(img)
    return img


def process_class(cls):
    src_folder = os.path.join(DATASET_DIR, cls)
    files = sorted([f for f in os.listdir(src_folder) if f.endswith(".jpg")])

    if not files:
        print(f"  WARNING: no images found in {src_folder}")
        return

    random.shuffle(files)
    split_idx = int(len(files) * TRAIN_SPLIT)
    splits = {
        "training": files[:split_idx],
        "testing":  files[split_idx:],
    }

    for split_name, split_files in splits.items():
        out_folder = os.path.join(OUTPUT_DIR, cls, split_name)
        saved = 0
        for fname in split_files:
            img = cv2.imread(os.path.join(src_folder, fname))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            stem = os.path.splitext(fname)[0]

            # Save the original (resized)
            cv2.imwrite(os.path.join(out_folder, f"{stem}_orig.jpg"), img)
            saved += 1

            # Save augmented copies
            for i in range(AUGMENTS_PER_IMAGE):
                aug = augment(img.copy())
                cv2.imwrite(os.path.join(out_folder, f"{stem}_aug{i}.jpg"), aug)
                saved += 1

        print(f"  [{cls}] {split_name:8s}: {len(split_files):4d} originals  "
              f"→  {saved} images saved")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=== Dataset Augmentation ===")
    print(f"Source      : {DATASET_DIR}/")
    print(f"Output      : {OUTPUT_DIR}/")
    print(f"Train split : {int(TRAIN_SPLIT * 100)} %  |  "
          f"Test split : {int((1 - TRAIN_SPLIT) * 100)} %")
    print(f"Augments per image : {AUGMENTS_PER_IMAGE}  "
          f"({AUGMENTS_PER_IMAGE + 1} total copies each)\n")

    ensure_dirs()

    for cls in CLASSES:
        process_class(cls)

    print("\nDone. Output structure:")
    for cls in CLASSES:
        for split in ("training", "testing"):
            folder = os.path.join(OUTPUT_DIR, cls, split)
            count  = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
            print(f"  {folder:<45} {count} images")


if __name__ == "__main__":
    main()
