import sys
sys.path.append('../src')

import pytest
from detector import ObjectDetector


@pytest.fixture(scope='module')
def detector():
    return ObjectDetector(model_size='n')


def test_model_loads(detector):
    assert detector.model is not None


def test_model_info(detector):
    info = detector.get_model_info()
    assert info['classes'] == 80
    assert 'person' in info['class_names']


def test_detect_from_url(detector):
    detections = detector.detect_from_url(
        'https://ultralytics.com/images/bus.jpg')
    assert len(detections) > 0


def test_detection_format(detector):
    detections = detector.detect_from_url(
        'https://ultralytics.com/images/bus.jpg')
    for d in detections:
        assert 'class' in d
        assert 'confidence' in d
        assert 'bbox' in d
        assert 0 <= d['confidence'] <= 1
        assert len(d['bbox']) == 4


def test_detect_persons(detector):
    detections = detector.detect_from_url(
        'https://ultralytics.com/images/bus.jpg')
    classes = [d['class'] for d in detections]
    assert 'person' in classes


def test_detect_bus(detector):
    detections = detector.detect_from_url(
        'https://ultralytics.com/images/bus.jpg')
    classes = [d['class'] for d in detections]
    assert 'bus' in classes


def test_confidence_threshold(detector):
    high_conf = detector.detect_from_url(
        'https://ultralytics.com/images/bus.jpg', conf=0.8)
    low_conf = detector.detect_from_url(
        'https://ultralytics.com/images/bus.jpg', conf=0.1)
    assert len(low_conf) >= len(high_conf)


def test_get_class_counts(detector):
    detections = detector.detect_from_url(
        'https://ultralytics.com/images/bus.jpg')
    counts = detector.get_class_counts(detections)
    assert isinstance(counts, dict)
    assert counts.get('person', 0) >= 1