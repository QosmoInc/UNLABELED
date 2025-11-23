"""Image detection - single image and batch processing."""

from typing import Optional, List
from pathlib import Path
import glob

from PIL import Image

from .base_detector import BaseDetector


class ImageDetector(BaseDetector):
    """Detector for single image or batch image processing."""

    def detect_image_file(
        self,
        imgfile: str,
        save_path: Optional[str] = None
    ) -> List:
        """Detect objects in a single image file.

        Args:
            imgfile: Path to input image
            save_path: Optional path to save visualization (defaults to input path)

        Returns:
            List of detection boxes
        """
        if save_path is None:
            save_path = imgfile

        img = Image.open(imgfile).convert('RGB')
        _, boxes = self.detect_and_visualize_pil(img, save_path)

        if self.verbose:
            print(f'{imgfile}: Detected {len(boxes)} objects')

        return boxes

    def detect_directory(
        self,
        img_dir: str,
        pattern: str = '*.png',
        save_dir: Optional[str] = None
    ) -> dict:
        """Detect objects in all images in a directory.

        Args:
            img_dir: Directory containing images
            pattern: Glob pattern for image files (default: '*.png')
            save_dir: Optional directory to save visualizations (defaults to img_dir)

        Returns:
            Dictionary mapping image filenames to detection boxes
        """
        img_files = glob.glob(f'{img_dir}/{pattern}')

        if self.verbose:
            print(f'Found {len(img_files)} images in {img_dir}')

        results = {}
        for imgfile in img_files:
            if save_dir is not None:
                filename = Path(imgfile).name
                save_path = str(Path(save_dir) / filename)
            else:
                save_path = imgfile

            boxes = self.detect_image_file(imgfile, save_path)
            results[imgfile] = boxes

        return results
