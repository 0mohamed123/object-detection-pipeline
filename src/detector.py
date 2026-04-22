from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class ObjectDetector:
    def __init__(self, model_size='n'):
        """
        model_size: 'n' (nano), 's' (small), 'm' (medium)
        """
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.model_size = model_size

    def detect_image(self, image_path, conf=0.25, save=False, output_dir='results'):
        results = self.model(image_path, conf=conf)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    'class': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
            if save:
                Path(output_dir).mkdir(exist_ok=True)
                r.save(filename=f"{output_dir}/detected_{Path(image_path).name}")
        return detections

    def detect_from_url(self, url, conf=0.25):
        results = self.model(url, conf=conf)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    'class': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
        return detections

    def detect_batch(self, image_paths, conf=0.25):
        all_detections = []
        for path in image_paths:
            detections = self.detect_image(path, conf=conf)
            all_detections.append({
                'image': path,
                'detections': detections,
                'count': len(detections)
            })
        return all_detections

    def get_class_counts(self, detections):
        counts = {}
        for d in detections:
            cls = d['class']
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def get_model_info(self):
        return {
            'model': f'YOLOv8{self.model_size}',
            'classes': len(self.model.names),
            'class_names': list(self.model.names.values())[:10]
        }