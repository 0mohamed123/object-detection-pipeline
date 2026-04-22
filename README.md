# Object Detection Pipeline (YOLOv8)

![Language](https://img.shields.io/badge/Language-Python-blue)
![Model](https://img.shields.io/badge/Model-YOLOv8-purple)
![Tests](https://img.shields.io/badge/Tests-8%20passing-green)
![Classes](https://img.shields.io/badge/Classes-80-orange)

Real-time object detection pipeline using YOLOv8 capable of detecting
80 object classes with confidence scores and bounding boxes.

## Detection Results

    Image: bus.jpg
    Detections: 6 objects
      bus        : 1 object | avg conf: 0.87
      person     : 4 objects| avg conf: 0.70
      stop sign  : 1 object | avg conf: 0.26

    Image: zidane.jpg
    Detections: 3 objects
      person     : 2 objects| avg conf: 0.83
      tie        : 1 object | avg conf: 0.29

    Total detections: 9 | Avg time per image: 3.80s

## Quick Start

    git clone https://github.com/0mohamed123/object-detection-pipeline.git
    cd object-detection-pipeline
    pip install ultralytics opencv-python

    # Run evaluation
    cd src
    python evaluate.py

    # Run tests
    cd ../tests
    python -m pytest test_detector.py -v

## Features

- Single image detection with confidence threshold control
- Batch detection on multiple images
- URL-based detection
- Class count aggregation
- Bounding box visualization with OpenCV
- Supports YOLOv8 nano/small/medium models

## Test Results

    8 passed | 0 failed

    Tests cover: model loading, model info, URL detection,
    detection format, person detection, bus detection,
    confidence threshold filtering, class counting

## Project Structure

    object-detection-pipeline/
    ├── src/
    │   ├── detector.py    # Core detection class
    │   ├── evaluate.py    # Evaluation pipeline
    │   └── visualize.py   # Bounding box visualization
    └── tests/
        └── test_detector.py  # 8 automated tests

## Technologies

- Python 3.12
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- pytest (8 tests)