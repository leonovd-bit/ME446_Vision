"""
collect_data.py — Automated training image capture for shape classification.

Usage:
    python collect_data.py

Controls:
    q  — quit early
    p  — pause / resume

Images are saved to:
    dataset/circle/circle_XXXX.jpg
    dataset/square/square_XXXX.jpg
"""

import os
import time
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"
ALL_CLASSES = {"c": "circle", "s": "square"}
CAPTURE_INTERVAL = 5        # countdown seconds shown between each photo
COUNTDOWN_SECONDS = 5
INITIAL_COUNTDOWN = 10     # countdown before the very first photo of each class
CAMERA_INDEX = 0            # 0 = default/built-in webcam
CAMERA_WIDTH  = 640         # standard resolution
CAMERA_HEIGHT = 480
CROP_PADDING = 40           # extra pixels to include around the detected object
WARP_SIZE    = 400          # 400 px = 400 mm → 1 px per mm top-down output
# ---------------------------------------------------------------------------

# ArUco detector (shared across all calls)
_ARUCO_DICT   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)


def detect_and_warp(frame):
    """Warp *frame* to a top-down 400x400 mm view using ArUco corner markers 0-3.

    Marker placement:
        ID 0 – top-left       → maps to (  0,   0) mm
        ID 1 – top-right      → maps to (400,   0) mm
        ID 2 – bottom-right   → maps to (400, 400) mm
        ID 3 – bottom-left    → maps to (  0, 400) mm

    Uses the center of each marker as the source point and findHomography
    for a robust perspective transform.
    Returns (warped, True) if all 4 markers found, else (original, False).
    """
    corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(frame)

    if ids is None or len(ids) < 4:
        return frame, False

    id_to_center = {}
    for i in range(len(ids)):
        marker_id = int(ids[i][0])
        # Center = mean of the 4 corner points of this marker
        center = corners[i][0].mean(axis=0)
        id_to_center[marker_id] = center

    if not all(k in id_to_center for k in (0, 1, 2, 3)):
        return frame, False

    src = np.array([
        id_to_center[0],   # top-left
        id_to_center[1],   # top-right
        id_to_center[2],   # bottom-right
        id_to_center[3],   # bottom-left
    ], dtype=np.float32)

    dst = np.array([
        [0,         0        ],
        [WARP_SIZE, 0        ],
        [WARP_SIZE, WARP_SIZE],
        [0,         WARP_SIZE],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
    return warped, True


def ensure_dirs(classes):
    """Create dataset sub-directories and clear any existing images."""
    for cls in classes:
        path = os.path.join(DATASET_DIR, cls)
        os.makedirs(path, exist_ok=True)
        deleted = 0
        for f in os.listdir(path):
            if f.endswith(".jpg"):
                os.remove(os.path.join(path, f))
                deleted += 1
        if deleted:
            print(f"  Cleared {deleted} old image(s) from {path}")


def existing_count(cls):
    """Return the number of images already saved for *cls*."""
    path = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith(".jpg")])


def crop_to_object(frame):
    """Return a crop of *frame* tightly around the contour closest to the center.

    Strategy:
      1. Convert to greyscale and blur to kill noise.
      2. Otsu threshold to separate object from background.
      3. Find contours and pick the one whose center is nearest the frame center.
      4. Expand the bounding box by CROP_PADDING and clamp to frame edges.

    Falls back to the original frame if no contour is found.
    """
    fh, fw = frame.shape[:2]
    frame_cx, frame_cy = fw / 2, fh / 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame  # nothing found — save full frame

    def dist_to_center(c):
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        return (cx - frame_cx) ** 2 + (cy - frame_cy) ** 2

    closest = min(contours, key=dist_to_center)
    x, y, w, h = cv2.boundingRect(closest)

    # Add padding, clamped to frame dimensions
    fh, fw = frame.shape[:2]
    x1 = max(x - CROP_PADDING, 0)
    y1 = max(y - CROP_PADDING, 0)
    x2 = min(x + w + CROP_PADDING, fw)
    y2 = min(y + h + CROP_PADDING, fh)

    return frame[y1:y2, x1:x2]


def _check_key(cap):
    """Read a key and handle pause. Returns 'q' to quit, 'p' if paused, None otherwise."""
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        return "q"
    if key == ord("p"):
        # Paused — show overlay and wait for 'p' again
        while True:
            ret, frame = cap.read()
            if ret:
                frame, _ = detect_and_warp(frame)
                overlay = frame.copy()
                cv2.putText(
                    overlay, "PAUSED - press P to resume",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )
                cv2.imshow("Collect Data", overlay)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("p"):
                return "p"
            if k == ord("q"):
                return "q"
    return None


def countdown(cap, seconds, message="Starting in", count=None, total=None):
    """Display a live countdown overlay. Returns False if 'q' is pressed."""
    for i in range(seconds, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            frame, warped_ok = detect_and_warp(frame)
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"{message} {i}...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            # Picture counter
            if count is not None and total is not None:
                cv2.putText(
                    overlay,
                    f"Photo {count}/{total}",
                    (10, overlay.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow("Collect Data", overlay)
            result = _check_key(cap)
            if result == "q":
                return False
            # After a pause the deadline may have passed — reset it
            if result == "p":
                deadline = time.time() + 1.0
    return True


def collect_class(cap, cls, images_per_class, session_files):
    """Capture up to images_per_class images for *cls*.

    Shows a countdown before each photo is taken.
    Appends every saved filepath to *session_files*.
    Returns False if the user pressed 'q' to quit early.
    """
    count = existing_count(cls)

    print(f"\n[{cls.upper()}] Starting from image {count}. "
          f"Target: {images_per_class}. Press 'q' to quit.")

    # Initial 10s countdown before first shot
    if not countdown(cap, INITIAL_COUNTDOWN, message="Get ready! Starting in",
                     count=count, total=images_per_class):
        return False

    while count < images_per_class:
        # --- Take the photo ---
        ret, frame = cap.read()
        if not ret:
            continue

        frame, _ = detect_and_warp(frame)
        cropped = crop_to_object(frame)
        filename = f"{cls}_{count:04d}.jpg"
        filepath = os.path.join(DATASET_DIR, cls, filename)
        cv2.imwrite(filepath, cropped)
        session_files.append(filepath)
        count += 1
        print(f"  Saved {filepath}  ({count}/{images_per_class})")

        if count >= images_per_class:
            break

        # --- Countdown to next photo, showing updated counter ---
        if not countdown(cap, CAPTURE_INTERVAL, message="Next photo in",
                         count=count, total=images_per_class):
            return False

    return True


def main():
    # --- Ask which shapes to collect ---
    print("=== Shape Data Collection ===")
    print("Which shape(s) do you want to collect?")
    print("  c  = circle")
    print("  s  = square")
    print("  cs = both")
    while True:
        choice = input("Enter choice (c / s / cs): ").strip().lower()
        classes = [ALL_CLASSES[ch] for ch in choice if ch in ALL_CLASSES]
        if classes:
            break
        print("  Invalid — please enter c, s, or cs.")

    # --- Ask how many images per shape ---
    while True:
        try:
            images_per_class = int(input("How many pictures per shape? ").strip())
            if images_per_class > 0:
                break
        except ValueError:
            pass
        print("  Please enter a positive whole number.")

    ensure_dirs(classes)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera. "
              "Try changing CAMERA_INDEX (0, 1, 2 …).")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")
    print(f"Shapes: {', '.join(classes)}  |  {images_per_class} images each  "
          f"({len(classes) * images_per_class} total)")

    session_files = []

    for i, cls in enumerate(classes):
        if i > 0:
            input(f"\nSwap shape to [{cls.upper()}] and press ENTER to continue…")

        ok = collect_class(cap, cls, images_per_class, session_files)
        if not ok:
            print("Quitting early — deleting all images captured this session…")
            for f in session_files:
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass
            print(f"  Deleted {len(session_files)} image(s).")
            break
        print(f"[{cls.upper()}] Done — {existing_count(cls)} images saved.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nCollection complete.")
    for cls in classes:
        print(f"  {cls}: {existing_count(cls)} images")


if __name__ == "__main__":
    main()
