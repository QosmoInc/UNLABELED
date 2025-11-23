"""Object detection evaluation tools.

This package provides unified interfaces for object detection across
different input sources (images, videos, cameras).
"""

from .base_detector import BaseDetector
from .image_detector import ImageDetector
from .video_detector import VideoDetector

__all__ = [
    'BaseDetector',
    'ImageDetector',
    'VideoDetector',
]
