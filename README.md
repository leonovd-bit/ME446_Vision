# ArUco Marker Detection, Object Localization, and Shape Classification

A computer vision project that uses ArUco markers to establish a calibrated reference plane, detect black objects within a 400×400mm workspace, and classify their shape (circle vs. square) using a trained CNN. The system performs perspective transformation to convert angled camera views into top-down coordinates with real-world spatial accuracy.

## Project Overview

This project implements a vision-based object detection and localization system using the following workflow:

1. **Marker Generation** - Create four ArUco markers to define the corners of a 400×400mm plane
2. **Image Capture** - Capture frames from a USB webcam
3. **Marker Detection** - Detect ArUco markers in the captured image
4. **Perspective Transformation** - Calculate homography matrix to convert pixel coordinates to real-world (mm) coordinates
5. **Object Detection** - Identify black objects in the warped image using binary thresholding
6. **Centroid Calculation** - Calculate the 2D and 3D centroid positions of detected objects

## System Requirements

- **Python 3.10+**
- **OpenCV** (cv2) - Computer vision library
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **TensorFlow 2.x** - CNN training (required for `augment_and_train.py`)
- **USB Webcam** - For image capture

## Installation

### Using Conda (Recommended)

```bash
conda create -n ME446_Vision python=3.10
conda activate ME446_Vision
pip install opencv-python numpy matplotlib tensorflow
```

### Using pip

```bash
pip install opencv-python numpy matplotlib tensorflow
```

## Quick Start

### 1. Generate ArUco Markers

Run cell 1 to generate four 200×200 pixel ArUco markers. These will be saved as:
- `aruco_marker_0.png` (Top-left)
- `aruco_marker_1.png` (Top-right)
- `aruco_marker_2.png` (Bottom-right)
- `aruco_marker_3.png` (Bottom-left)

**Print these markers** and arrange them at the corners of your 400×400mm workspace.

### 2. Capture Reference Image

Run cell 2 to:
- Initialize the USB webcam
- Display a live preview
- **Press SPACEBAR** to capture an image
- **Press 'Q'** to cancel without capturing

The captured image will be saved as `captured_frame.jpg`.

### 3. Detect Markers & Calculate Transformation

Run cells 3-4 to initialize the ArUco detector, then run cell 5 to:
- Detect all four markers in the image
- Calculate marker positions in pixel coordinates
- Display detected marker locations
- Verify all 4 markers were found

### 4. Apply Perspective Transformation

Run cell 6 to calculate the homography matrix, then cell 7 to:
- Warp the image to a top-down view
- Create a 400×400 pixel output where 1 pixel = 1 mm

### 5. Detect Black Objects

Run cells 8-9 to:
- Apply binary thresholding to detect black pixels
- Apply morphological operations to clean up noise
- Find contours in the binary image

### 6. Localize Object Centroid

Run cell 10 to:
- Calculate the 2D centroid (X, Y) of the black object
- Visualize the detection with green outline and red crosshair
- Display coordinates in real-world mm units

### 7. Calculate 3D Centroid (Optional)

Run cell 11 to calculate the true 3D centroid position, accounting for the block height (25.4mm for a 1×1×1 inch cube).

## Coordinate System

- **X-axis**: Horizontal (0 = left edge, 400 = right edge in mm)
- **Y-axis**: Vertical (0 = top edge, 400 = bottom edge in mm)
- **Z-axis**: Height above the plane (0 = plane surface, 12.7mm = centroid of 25.4mm block)
- **Origin**: Top-left corner of the plane

## Troubleshooting

### No markers detected
- Verify markers are visible and not occluded
- Check lighting - avoid harsh shadows
- Ensure all four markers are within the camera frame
- Try adjusting camera angle

### Object not detected
- Check that black object is properly placed on the plane
- Verify adequate lighting (no strong shadows)
- Adjust the `threshold_value` in cell 8 (increase to 80-100 if needed)
- Verify `min_area` threshold is appropriate for your object

### Camera not initializing
- Try changing `cv2.VideoCapture(0)` to index 1 or 2
- Verify USB webcam is connected and recognized
- Check for permission issues with camera access

### Poor perspective transformation accuracy
- Ensure all four markers are clearly visible
- Verify markers are at exact 400×400mm corners
- Check that camera view is not severely angled
- Re-capture image with better marker visibility

## Project Structure

```
ME446_Vision/
├── collect_data.py                  # Automated training image capture
├── augment_and_train.py             # Data augmentation + CNN training
├── ArUco_detection_mapping.ipynb    # Main notebook with all steps
├── README.md                         # This file
├── aruco_marker_0.png               # Generated marker (top-left)
├── aruco_marker_1.png               # Generated marker (top-right)
├── aruco_marker_2.png               # Generated marker (bottom-right)
├── aruco_marker_3.png               # Generated marker (bottom-left)
├── captured_frame.jpg               # Captured webcam image (generated at runtime)
├── dataset/                          # Created at runtime by collect_data.py
│   ├── circle/
│   └── square/
└── best_model.keras                  # Best trained model (generated at runtime)
```

## Configuration

Key parameters you can adjust in the notebook:

| Parameter | Cell | Default | Purpose |
|-----------|------|---------|---------|
| `marker_size` | 1 | 200 px | Size of generated ArUco markers |
| `threshold_value` | 8 | 50 | Threshold for black object detection |
| `min_area` | 9 | 100 mm² | Minimum object area to detect |
| `block_height_mm` | 11 | 25.4 | Height of object for 3D centroid calculation |

## Output Information

### 2D Detection (Cell 9)
```
Block Centroid Location:
   X = 150.42 mm
   Y = 200.18 mm
   Area = 2547.50 mm²
```

### 3D Centroid (Cell 11)
```
Block 3D Centroid (accounting for height):
   X = 150.42 mm
   Y = 200.18 mm
   Z = 12.70 mm (above the plane)
```

## Shape Classification — Data Collection & Training

### Overview

`collect_data.py` captures training images using a USB webcam under a consistent camera stand, background, and lighting. `augment_and_train.py` augments those images and trains a CNN to classify shapes as **circle** or **square**.

---

### Step A — Collect Training Images

```bash
python collect_data.py
```

**What it does:**

1. Creates `dataset/circle/` and `dataset/square/` directories.
2. Counts any images already present (safe to restart mid-collection).
3. Starts a 3-second countdown, then captures a frame every **2 seconds** automatically — you reposition/rotate the physical shape between captures.
4. Shows a live preview window with a status overlay (class name + progress).
5. After 200 circle images, prompts you to swap the shape, then collects 200 square images.

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `p` | Pause / resume capture |
| `q` | Quit early |

**Configuration** (top of `collect_data.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `IMAGES_PER_CLASS` | 200 | Images to collect per class |
| `CAPTURE_INTERVAL` | 2 s | Seconds between automatic captures |
| `COUNTDOWN_SECONDS` | 3 | Pre-capture countdown |
| `CAMERA_INDEX` | 0 | USB camera index |
| `CAMERA_WIDTH/HEIGHT` | 640×480 | Camera resolution |

---

### Step B — Train the CNN

```bash
python augment_and_train.py
```

**Augmentation strategy**

Designed for a consistent background and lighting setup — only augments what genuinely varies during robot arm inference:

| Augmentation | Value | Rationale |
|---|---|---|
| Rotation | ±45° | Shape can be oriented differently |
| Width / height shift | ±20% | Shape won't always be centred |
| Zoom | ±20% | Robot arm distance to shape varies |
| Shear | ±10% | Slight perspective distortion |
| Brightness | 85%–115% | Minor lighting drift |
| Fill mode | `constant`, `cval=255` | Preserves white background |
| Flips, heavy colour jitter | **disabled** | Not needed for this use case |

- **Validation split**: 80% training / 20% validation
- **Image size**: 128×128 px, grayscale, normalised to [0, 1]

**CNN Architecture**

```
Input (128×128×1)
│
├── Conv2D(32) → BatchNorm → MaxPool(2×2)
├── Conv2D(64) → BatchNorm → MaxPool(2×2)
└── Conv2D(128) → BatchNorm → MaxPool(2×2)
    │
    Flatten → Dropout(0.5) → Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)
```

- **Loss**: binary crossentropy
- **Optimizer**: Adam
- **EarlyStopping**: monitors `val_accuracy`, patience = 10, restores best weights
- **ModelCheckpoint**: saves best model to `best_model.keras`
- **Max epochs**: 50

**Configuration** (top of `augment_and_train.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATASET_DIR` | `dataset` | Path to collected images |
| `IMG_SIZE` | (128, 128) | Resize target |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 50 | Maximum training epochs |
| `MODEL_PATH` | `best_model.keras` | Output model file |

---

## References

- **ArUco Markers**: [OpenCV ArUco Documentation](https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html)
- **Perspective Transformation**: [Homography Matrix](https://en.wikipedia.org/wiki/Homography_(computer_vision))
- **Contour Detection**: [OpenCV Contours](https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html)

## License

This project is for educational purposes.

## Author

Ronald Vasquez, Salwa Omar, Davyd Leonovets

---

**Last Updated**: March 2026
