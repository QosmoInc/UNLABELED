"""Base detector class for object detection evaluation.

This module provides a common base class for all detection scripts,
eliminating code duplication and providing a consistent interface.
"""

from typing import Optional, List, Tuple
from pathlib import Path
import time

import torch
from PIL import Image
import numpy as np

from darknet import Darknet
from utils import do_detect, load_class_names, plot_boxes


class BaseDetector:
    """Base class for object detection using YOLO models.

    This class handles common functionality like model loading, preprocessing,
    and detection. Subclasses can override specific methods for different
    detection modes (image, video, camera, etc.).

    Attributes:
        model: Loaded Darknet model
        class_names: List of class names from COCO/VOC dataset
        use_cuda: Whether to use CUDA for inference
        conf_thresh: Confidence threshold for detections
        nms_thresh: NMS threshold for detections
    """

    def __init__(
        self,
        cfgfile: str,
        weightfile: str,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        use_cuda: bool = True,
        verbose: bool = True
    ) -> None:
        """Initialize the detector.

        Args:
            cfgfile: Path to YOLO configuration file
            weightfile: Path to YOLO weights file
            conf_thresh: Confidence threshold for detections (0.0-1.0)
            nms_thresh: Non-maximum suppression threshold (0.0-1.0)
            use_cuda: Whether to use CUDA acceleration if available
            verbose: Whether to print detailed information
        """
        self.verbose = verbose
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Load model
        self.model = Darknet(cfgfile)
        if self.verbose:
            self.model.print_network()

        self.model.load_weights(weightfile)
        if self.verbose:
            print(f'Loading weights from {weightfile}... Done!')

        # Set device
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()

        # Load class names
        if self.model.num_classes == 20:
            namesfile = 'data/voc.names'
        elif self.model.num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'

        self.class_names = load_class_names(namesfile)

    def preprocess_pil_image(self, img: Image.Image) -> Image.Image:
        """Preprocess PIL image for detection.

        Args:
            img: Input PIL Image

        Returns:
            Resized image matching model input dimensions
        """
        return img.resize((self.model.width, self.model.height))

    def preprocess_numpy_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess numpy/cv2 image for detection.

        Args:
            img: Input numpy array (BGR format from cv2)

        Returns:
            Resized and color-converted image (RGB format)
        """
        import cv2
        sized = cv2.resize(img, (self.model.width, self.model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        return sized

    def detect_pil(
        self,
        img: Image.Image,
        measure_time: bool = True
    ) -> Tuple[List, Optional[float]]:
        """Run detection on a PIL image.

        Args:
            img: Input PIL Image
            measure_time: Whether to measure inference time

        Returns:
            Tuple of (boxes, inference_time). boxes is a list of detections,
            inference_time is in seconds (None if measure_time=False)
        """
        sized = self.preprocess_pil_image(img)

        start_time = time.time() if measure_time else None
        boxes = do_detect(self.model, sized, self.conf_thresh, self.nms_thresh, self.use_cuda)
        inference_time = (time.time() - start_time) if measure_time else None

        return boxes, inference_time

    def detect_numpy(
        self,
        img: np.ndarray,
        measure_time: bool = True
    ) -> Tuple[List, Optional[float]]:
        """Run detection on a numpy/cv2 image.

        Args:
            img: Input numpy array (BGR format from cv2)
            measure_time: Whether to measure inference time

        Returns:
            Tuple of (boxes, inference_time). boxes is a list of detections,
            inference_time is in seconds (None if measure_time=False)
        """
        sized = self.preprocess_numpy_image(img)

        start_time = time.time() if measure_time else None
        boxes = do_detect(self.model, sized, self.conf_thresh, self.nms_thresh, self.use_cuda)
        inference_time = (time.time() - start_time) if measure_time else None

        return boxes, inference_time

    def detect_and_visualize_pil(
        self,
        img: Image.Image,
        save_path: Optional[str] = None
    ) -> Tuple[Image.Image, List]:
        """Detect objects and visualize results on PIL image.

        Args:
            img: Input PIL Image
            save_path: Optional path to save the visualized image

        Returns:
            Tuple of (visualized_image, boxes)
        """
        boxes, inference_time = self.detect_pil(img, measure_time=True)

        if self.verbose and inference_time is not None:
            print(f'Predicted in {inference_time:.4f} seconds.')

        # Visualize boxes
        result_img = plot_boxes(img, boxes, save_path, self.class_names)

        return result_img, boxes

    def detect_and_visualize_numpy(
        self,
        img: np.ndarray,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, List]:
        """Detect objects and visualize results on numpy/cv2 image.

        Args:
            img: Input numpy array (BGR format from cv2)
            save_path: Optional path to save the visualized image

        Returns:
            Tuple of (visualized_image, boxes). Image is in RGB format.
        """
        boxes, inference_time = self.detect_numpy(img, measure_time=True)

        if self.verbose and inference_time is not None:
            print(f'Predicted in {inference_time:.4f} seconds.')

        # Convert to PIL for visualization, then back to numpy
        pil_img = Image.fromarray(img)
        result_pil = plot_boxes(pil_img, boxes, save_path, self.class_names)
        result_numpy = np.array(result_pil)

        return result_numpy, boxes
