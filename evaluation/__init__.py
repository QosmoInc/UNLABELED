"""Evaluation tools for adversarial patch detection.

This package provides tools for evaluating adversarial patches and
running object detection on images, videos, and camera streams.

Modules:
    - detectors: Object detection classes (BaseDetector, ImageDetector, VideoDetector)
    - patch_evaluator: Adversarial patch evaluation tools
    - detect_*: Command-line detection scripts
    - check_recognition_rate: Patch effectiveness measurement
"""

from .detectors import BaseDetector, ImageDetector, VideoDetector
from .patch_evaluator import PatchEvaluator

__all__ = [
    'BaseDetector',
    'ImageDetector',
    'VideoDetector',
    'PatchEvaluator',
]
