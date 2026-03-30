"""
classify.py — Capture an image from the webcam and classify it as circle or square.

Uses the same ArUco-based perspective warp and object cropping as collect_data.py.

Usage:
    python classify.py

Controls:
    SPACEBAR — capture and classify
    q        — quit
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "shape_classifier.pth"
CLASSES = ["circle", "square"]
IMG_SIZE = 128
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WARP_SIZE = 400
CROP_PADDING = 40
# ---------------------------------------------------------------------------

# ArUco detector
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)


def detect_and_warp(frame):
    """Warp *frame* to a top-down 400x400 mm view using ArUco corner markers 0-3.

    Returns (warped, True) if all 4 markers found, else (original, False).
    """
    corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(frame)

    if ids is None or len(ids) < 4:
        return frame, False

    id_to_center = {}
    for i in range(len(ids)):
        marker_id = int(ids[i][0])
        center = corners[i][0].mean(axis=0)
        id_to_center[marker_id] = center

    if not all(k in id_to_center for k in (0, 1, 2, 3)):
        return frame, False

    src = np.array([
        id_to_center[0], id_to_center[1],
        id_to_center[2], id_to_center[3],
    ], dtype=np.float32)

    dst = np.array([
        [0, 0], [WARP_SIZE, 0],
        [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
    return warped, True


def crop_to_object(frame):
    """Return a crop tightly around the contour closest to the frame center."""
    fh, fw = frame.shape[:2]
    frame_cx, frame_cy = fw / 2, fh / 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame

    def dist_to_center(c):
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        return (cx - frame_cx) ** 2 + (cy - frame_cy) ** 2

    closest = min(contours, key=dist_to_center)
    x, y, w, h = cv2.boundingRect(closest)

    x1 = max(x - CROP_PADDING, 0)
    y1 = max(y - CROP_PADDING, 0)
    x2 = min(x + w + CROP_PADDING, fw)
    y2 = min(y + h + CROP_PADDING, fh)

    return frame[y1:y2, x1:x2]


class ShapeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model():
    model = ShapeCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def preprocess(frame):
    """Warp, crop, resize and convert a BGR frame to a model-ready tensor."""
    warped, ok = detect_and_warp(frame)
    cropped = crop_to_object(warped)
    img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img).unsqueeze(0), ok  # (1, 3, H, W)


def main():
    model = load_model()
    print(f"Model loaded from {MODEL_PATH}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Press SPACEBAR to capture & classify, 'q' to quit.")

    result_text = ""
    result_color = (255, 255, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Show warped view as live preview
        display, warped_ok = detect_and_warp(frame)
        display = display.copy()

        if not warped_ok:
            cv2.putText(display, "ArUco markers not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if result_text:
            cv2.putText(display, result_text, (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2, cv2.LINE_AA)

        cv2.imshow("Classify - SPACE to capture, Q to quit", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:  # spacebar
            tensor, ok = preprocess(frame)
            if not ok:
                result_text = "No ArUco markers - try again"
                result_color = (0, 0, 255)
                print("  -> ArUco markers not detected, cannot classify")
                continue

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()

            label = CLASSES[pred_idx]
            result_text = f"{label} ({confidence:.1%})"
            result_color = (0, 255, 0) if confidence > 0.7 else (0, 200, 255)
            print(f"  -> {label}  confidence={confidence:.1%}")

    cap.release()
    cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
