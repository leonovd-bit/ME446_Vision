"""
train_cnn.py — Train a PyTorch CNN to classify circle vs square.

Usage:
    python train_cnn.py

Reads training images from:
    augmented_dataset/circle/training/
    augmented_dataset/square/training/

Saves:
    shape_classifier.pth   — trained model weights
    training_plot.png      — accuracy / loss curves
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "augmented_dataset"
CLASSES = ["circle", "square"]       # label 0 = circle, 1 = square
IMG_SIZE = 128                       # resize all images to 128x128
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_PATH = "shape_classifier.pth"
PLOT_PATH = "training_plot.png"
SEED = 42
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)


class ShapeDataset(Dataset):
    """Loads .jpg images from augmented_dataset/<class>/<split>/."""

    def __init__(self, data_dir, classes, img_size, split="training"):
        self.samples = []  # list of (filepath, label_index)
        self.img_size = img_size

        for label_idx, cls in enumerate(classes):
            folder = os.path.join(data_dir, cls, split)
            if not os.path.isdir(folder):
                print(f"WARNING: {folder} not found — skipping")
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(".jpg"):
                    self.samples.append((os.path.join(folder, fname), label_idx))

        print(f"Loaded {len(self.samples)} {split} images  "
              f"({', '.join(f'{c}: {sum(1 for s in self.samples if s[1]==i)}' for i, c in enumerate(classes))})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        # HWC uint8 → CHW float32, normalised to [0, 1]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        return torch.from_numpy(img), label


class ShapeCNN(nn.Module):
    """Simple 3-block CNN for binary classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After 3 pools: 128 -> 64 -> 32 -> 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset & loaders ---
    train_ds = ShapeDataset(DATA_DIR, CLASSES, IMG_SIZE, split="training")
    test_ds = ShapeDataset(DATA_DIR, CLASSES, IMG_SIZE, split="testing")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print(f"Train: {len(train_ds)}  |  Test: {len(test_ds)}")

    # --- Model, loss, optimiser ---
    model = ShapeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    # --- Training loop ---
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_test_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # -- Train --
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), torch.tensor(labels, dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # -- Test --
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), torch.tensor(labels, dtype=torch.long, device=device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += imgs.size(0)

        test_loss = running_loss / total
        test_acc = correct / total
        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.3f}  lr={lr:.1e}")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest test accuracy: {best_test_acc:.3f}")
    print(f"Model saved to {MODEL_PATH}")

    # --- Plot ---
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, history["train_loss"], label="Train")
    ax1.plot(epochs_range, history["test_loss"], label="Test")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["train_acc"], label="Train")
    ax2.plot(epochs_range, history["test_acc"], label="Test")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Training plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    train()
