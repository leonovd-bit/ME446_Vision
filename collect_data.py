"""
collect_data.py — Interactive training image capture for shape classification.

Usage:
    python collect_data.py

Controls:
    c  — capture current frame as a circle image
    s  — capture current frame as a square image
    q  — quit

Images are saved to:
    dataset/circle/circle_XXXX.jpg
    dataset/square/square_XXXX.jpg
"""

import os
import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"
CLASSES = ["circle", "square"]
IMAGES_PER_CLASS = 200
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


def save_image(frame, cls, count):
    """Save *frame* as the next image for *cls* and return the new count."""
    filename = f"{cls}_{count:04d}.jpg"
    filepath = os.path.join(DATASET_DIR, cls, filename)
    cv2.imwrite(filepath, frame)
    count += 1
    print(f"  Saved {filepath}  ({count}/{IMAGES_PER_CLASS})")
    return count


def main():
    ensure_dirs()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera. "
              "Try changing CAMERA_INDEX (0, 1, 2 …).")
        return

    counts = {cls: existing_count(cls) for cls in CLASSES}

    print("=== Shape Data Collection ===")
    print(f"Target: {IMAGES_PER_CLASS} images per class  "
          f"({len(CLASSES) * IMAGES_PER_CLASS} total)")
    print("Press 'c' to capture circle, 's' to capture square, 'q' to quit.")

    while True:
        # Auto-quit when both classes are complete
        if all(counts[cls] >= IMAGES_PER_CLASS for cls in CLASSES):
            print("All classes complete.")
            break

        ret, frame = cap.read()
        if not ret:
            continue

        # Build overlay showing current counts for both classes
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"circle: {counts['circle']}/{IMAGES_PER_CLASS}  "
            f"square: {counts['square']}/{IMAGES_PER_CLASS}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Collect Data", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting early.")
            break
        elif key == ord("c"):
            if counts["circle"] < IMAGES_PER_CLASS:
                counts["circle"] = save_image(frame, "circle", counts["circle"])
            else:
                print("  circle: target already reached.")
        elif key == ord("s"):
            if counts["square"] < IMAGES_PER_CLASS:
                counts["square"] = save_image(frame, "square", counts["square"])
            else:
                print("  square: target already reached.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nCollection complete.")
    for cls in CLASSES:
        print(f"  {cls}: {existing_count(cls)} images")


if __name__ == "__main__":
    main()
