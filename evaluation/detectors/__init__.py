"""Object detection evaluation tools.

This package provides unified interfaces for object detection across
different input sources (images, videos, cameras).
"""

from .base_detector import BaseDetector
from .video_detector import VideoDetector

__all__ = [
    'BaseDetector',
    'VideoDetector',
]
