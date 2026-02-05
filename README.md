# ArUco Marker Detection and Object Localization

A computer vision project that uses ArUco markers to establish a calibrated reference plane and detect black objects within a 400×400mm workspace. The system performs perspective transformation to convert angled camera views into top-down coordinates with real-world spatial accuracy.

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
- **USB Webcam** - For image capture

## Installation

### Using Conda (Recommended)

```bash
conda create -n ME446_Vision python=3.10
conda activate ME446_Vision
pip install opencv-python numpy matplotlib
```

### Using pip

```bash
pip install opencv-python numpy matplotlib
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
├── ArUco_detection_mapping.ipynb    # Main notebook with all steps
├── README.md                         # This file
├── aruco_marker_0.png               # Generated marker (top-left)
├── aruco_marker_1.png               # Generated marker (top-right)
├── aruco_marker_2.png               # Generated marker (bottom-right)
├── aruco_marker_3.png               # Generated marker (bottom-left)
└── captured_frame.jpg               # Captured webcam image (generated at runtime)
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

## References

- **ArUco Markers**: [OpenCV ArUco Documentation](https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html)
- **Perspective Transformation**: [Homography Matrix](https://en.wikipedia.org/wiki/Homography_(computer_vision))
- **Contour Detection**: [OpenCV Contours](https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html)

## License

This project is for educational purposes.

## Author

Ronald Vasquez, Salwa Omar, Davyd Leonovets

---

**Last Updated**: February 2026
