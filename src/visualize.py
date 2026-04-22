from detector import ObjectDetector
import cv2
import numpy as np
import urllib.request


def draw_detections(image, detections):
    colors = {
        'person': (0, 255, 0),
        'car': (255, 0, 0),
        'bus': (0, 0, 255),
        'default': (255, 255, 0)
    }

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det['bbox']]
        cls = det['class']
        conf = det['confidence']
        color = colors.get(cls, colors['default'])

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


def visualize_url(url, output_path='results/output.jpg'):
    import urllib.request
    import numpy as np

    detector = ObjectDetector(model_size='n')
    detections = detector.detect_from_url(url)

    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    image = draw_detections(image, detections)
    cv2.imwrite(output_path, image)
    print(f"Saved to {output_path}")
    return detections