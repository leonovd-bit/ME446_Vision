"""
collect_data.py — Automated training image capture for shape classification.

Usage:
    python collect_data.py

Controls:
    p  — pause / resume capture
    q  — quit early

Images are saved to:
    dataset/circle/circle_XXXX.jpg
    dataset/square/square_XXXX.jpg
"""

import os
import time
import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"
CLASSES = ["circle", "square"]
IMAGES_PER_CLASS = 200
CAPTURE_INTERVAL = 2        # seconds between automatic captures
COUNTDOWN_SECONDS = 3
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# ---------------------------------------------------------------------------


def ensure_dirs():
    """Create dataset sub-directories if they don't already exist."""
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)


def existing_count(cls):
    """Return the number of images already saved for *cls*."""
    path = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith(".jpg")])


def countdown(cap, seconds):
    """Display a live countdown overlay before capture starts."""
    for i in range(seconds, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Starting in {i}...",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 200, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("Collect Data", overlay)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                return False
    return True


def collect_class(cap, cls):
    """Capture up to IMAGES_PER_CLASS images for *cls*.

    Returns False if the user pressed 'q' to quit early.
    """
    count = existing_count(cls)
    paused = False
    last_capture = time.time()

    print(f"\n[{cls.upper()}] Resuming from image {count}. "
          f"Target: {IMAGES_PER_CLASS}. Press 'p' to pause, 'q' to quit.")

    if not countdown(cap, COUNTDOWN_SECONDS):
        return False

    while count < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            continue

        # Build overlay text
        status = "PAUSED" if paused else "RECORDING"
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"{status}  [{cls}]  {count}/{IMAGES_PER_CLASS}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if paused else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Collect Data", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("p"):
            paused = not paused
            last_capture = time.time()  # reset timer after un-pause

        now = time.time()
        if not paused and (now - last_capture) >= CAPTURE_INTERVAL:
            filename = f"{cls}_{count:04d}.jpg"
            filepath = os.path.join(DATASET_DIR, cls, filename)
            cv2.imwrite(filepath, frame)
            count += 1
            last_capture = now
            print(f"  Saved {filepath}  ({count}/{IMAGES_PER_CLASS})")

    return True


def main():
    ensure_dirs()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera. "
              "Try changing CAMERA_INDEX (0, 1, 2 …).")
        return

    print("=== Shape Data Collection ===")
    print(f"Target: {IMAGES_PER_CLASS} images per class  "
          f"({len(CLASSES) * IMAGES_PER_CLASS} total)")

    for i, cls in enumerate(CLASSES):
        if i > 0:
            # Prompt the user to swap the physical shape
            input(f"\nSwap shape to [{cls.upper()}] and press ENTER to continue…")

        ok = collect_class(cap, cls)
        if not ok:
            print("Quitting early.")
            break
        print(f"[{cls.upper()}] Done — {existing_count(cls)} images saved.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nCollection complete.")
    for cls in CLASSES:
        print(f"  {cls}: {existing_count(cls)} images")


if __name__ == "__main__":
    main()
