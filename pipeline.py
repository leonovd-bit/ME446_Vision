"""
pipeline.py — Full vision pipeline: capture → ArUco warp → detect all objects
              → compute centroids → classify each as circle or square.

Usage:
    python pipeline.py

Controls:
    SPACEBAR — capture frame and run full pipeline
    q        — quit
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH      = "shape_classifier.pth"
CLASSES         = ["circle", "square"]
IMG_SIZE        = 128
CAMERA_INDEX    = 0
CAMERA_WIDTH    = 1280
CAMERA_HEIGHT   = 720
WARP_SIZE       = 400
CROP_PADDING    = 20
MIN_CONTOUR_AREA = 150   # px² — ignore specks smaller than this
ARUCO_PAD_PX    = 15     # dilation radius to mask ArUco edges before threshold
# ---------------------------------------------------------------------------

# ArUco detector
_ARUCO_DICT     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
_ARUCO_PARAMS   = cv2.aruco.DetectorParameters()
_ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)


# ---------------------------------------------------------------------------
# CNN model (same architecture as train_cnn.py / classify.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------
def detect_and_warp(frame):
    """Warp frame to a top-down WARP_SIZE x WARP_SIZE view via ArUco markers 0-3.

    Returns (warped, homography, True) on success, (frame, None, False) otherwise.
    """
    corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(frame)

    if ids is None or len(ids) < 4:
        return frame, None, False

    id_to_center = {}
    for i in range(len(ids)):
        marker_id = int(ids[i][0])
        id_to_center[marker_id] = corners[i][0].mean(axis=0)

    if not all(k in id_to_center for k in (0, 1, 2, 3)):
        return frame, None, False

    src = np.array([
        id_to_center[0], id_to_center[1],
        id_to_center[2], id_to_center[3],
    ], dtype=np.float32)

    dst = np.array([
        [0,         0        ],
        [WARP_SIZE, 0        ],
        [WARP_SIZE, WARP_SIZE],
        [0,         WARP_SIZE],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
    return warped, H, True


def build_aruco_mask(warped_gray, corners, H):
    """Return a uint8 mask (255 = valid, 0 = ArUco region) in the warped frame."""
    mask = np.ones_like(warped_gray, dtype=np.uint8) * 255
    h, w = warped_gray.shape

    for c in corners:
        pts = c[0].reshape(-1, 1, 2).astype(np.float32)
        warped_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        warped_pts = np.round(warped_pts).astype(np.int32)
        warped_pts[:, 0] = np.clip(warped_pts[:, 0], 0, w - 1)
        warped_pts[:, 1] = np.clip(warped_pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [warped_pts], 0)

    # Dilate exclusion zone so marker edges don't bleed into threshold
    exclusion = cv2.bitwise_not(mask)
    kernel = np.ones((ARUCO_PAD_PX, ARUCO_PAD_PX), np.uint8)
    exclusion = cv2.dilate(exclusion, kernel)
    return cv2.bitwise_not(exclusion)


def detect_objects(warped, corners, H):
    """Return list of (cx, cy, area, contour) for every object found in the warped image."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    mask = build_aruco_mask(gray, corners, H)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Otsu threshold (inverted: dark objects on light background → white blobs)
    _, binary = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological clean-up
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.bitwise_and(binary, binary, mask=mask)  # re-apply after morphology

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        bx, by, bw, bh = cv2.boundingRect(c)
        cy = int(by + 0.75 * bh)  # 3/4 down bounding box (targets object base)

        objects.append((cx, cy, area, c))

    return objects


def crop_object(warped, contour):
    """Crop the warped image tightly around a single contour for classification."""
    fh, fw = warped.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    x1 = max(x - CROP_PADDING, 0)
    y1 = max(y - CROP_PADDING, 0)
    x2 = min(x + w + CROP_PADDING, fw)
    y2 = min(y + h + CROP_PADDING, fh)
    return warped[y1:y2, x1:x2]


def classify_crop(model, crop):
    """Run the CNN on a single cropped BGR image. Returns (label, confidence)."""
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    return CLASSES[pred_idx], float(probs[pred_idx])


def draw_results(warped, objects, labels):
    """Draw contours, centroids, bounding boxes and labels onto a copy of warped."""
    out = warped.copy()

    for (cx, cy, area, contour), (label, conf) in zip(objects, labels):
        color = (0, 255, 0) if label == "circle" else (255, 100, 0)

        # Contour outline
        cv2.drawContours(out, [contour], -1, color, 2)

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(out, (x, y), (x + w, y + h), (128, 128, 128), 1)

        # Centroid dot
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)

        # Label + coordinates
        text = f"{label} {conf:.0%} ({cx},{cy}mm)"
        cv2.putText(out, text, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading model…")
    model = load_model()
    print(f"Model loaded from '{MODEL_PATH}'")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Press SPACEBAR to capture and classify, 'q' to quit.\n")

    result_frame = None  # last annotated result to keep showing

    # Re-detect ArUco each frame so we have corners available for masking
    last_corners = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        warped, H, ok = detect_and_warp(frame)

        if ok:
            # Update stored corners for the current frame
            raw_corners, raw_ids, _ = _ARUCO_DETECTOR.detectMarkers(frame)
            last_corners = raw_corners

            display = warped.copy()
            cv2.putText(display, "SPACE=capture  Q=quit", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            display = frame.copy()
            cv2.putText(display, "ArUco markers not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            last_corners = None

        cv2.imshow("Pipeline - live", display)

        if result_frame is not None:
            cv2.imshow("Pipeline - results", result_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:  # SPACEBAR
            if not ok or last_corners is None:
                print("  -> ArUco markers not detected, cannot run pipeline.")
                continue

            print("--- Running pipeline ---")

            # 1. Detect objects in the warped plane
            objects = detect_objects(warped, last_corners, H)
            print(f"  Objects found: {len(objects)}")

            if not objects:
                print("  -> No objects detected above minimum area.")
                continue

            # 2. Classify each object
            labels = []
            for i, (cx, cy, area, contour) in enumerate(objects):
                crop = crop_object(warped, contour)
                label, conf = classify_crop(model, crop)
                labels.append((label, conf))
                print(f"  Object {i+1}: centroid=({cx},{cy})mm  "
                      f"area={area:.0f}px²  -> {label} ({conf:.1%})")

            # 3. Draw and display results
            result_frame = draw_results(warped, objects, labels)
            cv2.imshow("Pipeline - results", result_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
