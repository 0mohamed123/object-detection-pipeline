from detector import ObjectDetector
import json
import time


def evaluate_on_samples():
    detector = ObjectDetector(model_size='n')

    test_urls = [
        'https://ultralytics.com/images/bus.jpg',
        'https://ultralytics.com/images/zidane.jpg',
    ]

    print("=" * 55)
    print("   Object Detection Pipeline - Evaluation")
    print("=" * 55)

    info = detector.get_model_info()
    print(f"\nModel: {info['model']}")
    print(f"Classes: {info['classes']}")

    total_detections = 0
    total_time = 0

    for url in test_urls:
        print(f"\nImage: {url.split('/')[-1]}")
        start = time.time()
        detections = detector.detect_from_url(url)
        elapsed = time.time() - start
        total_time += elapsed
        total_detections += len(detections)

        counts = detector.get_class_counts(detections)
        print(f"Detections: {len(detections)} objects in {elapsed:.2f}s")
        for cls, count in counts.items():
            confs = [d['confidence'] for d in detections if d['class'] == cls]
            avg_conf = sum(confs) / len(confs)
            print(f"  {cls:15s}: {count} object(s) | avg conf: {avg_conf:.2f}")

    print("\n" + "=" * 55)
    print(f"Total detections: {total_detections}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per image: {total_time/len(test_urls):.2f}s")
    print("=" * 55)


if __name__ == '__main__':
    evaluate_on_samples()