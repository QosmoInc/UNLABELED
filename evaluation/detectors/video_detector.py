"""Video detection - camera and video file processing."""

from typing import Optional, Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .base_detector import BaseDetector

# cv2 is imported lazily when VideoDetector is instantiated
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class VideoDetector(BaseDetector):
    """Detector for video files and camera streams.

    Requires opencv-python (cv2) to be installed.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize VideoDetector.

        Raises:
            ImportError: If opencv-python (cv2) is not installed
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "opencv-python (cv2) is required for VideoDetector. "
                "Install it with: uv add opencv-python"
            )
        super().__init__(*args, **kwargs)

    def detect_camera(
        self,
        camera_id: int = 0,
        display: bool = True,
        fullscreen: bool = True,
        callback: Optional[Callable[[np.ndarray, list], None]] = None
    ) -> None:
        """Detect objects from camera/webcam stream.

        Args:
            camera_id: Camera device ID (default: 0 for default camera)
            display: Whether to display results in window
            fullscreen: Whether to use fullscreen mode
            callback: Optional callback function(frame, boxes) called for each frame
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(f'Failed to open camera {camera_id}')

        if self.verbose:
            print(f'Camera {camera_id} opened. Press "q" to quit.')

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect objects
                result_img, boxes = self.detect_and_visualize_numpy(frame, save_path=None)

                # Call user callback if provided
                if callback is not None:
                    callback(result_img, boxes)

                # Display results
                if display:
                    if fullscreen:
                        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty('Result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow('Result', result_img)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

    def detect_video_file(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        display: bool = False,
        progress_bar: bool = True,
        callback: Optional[Callable[[np.ndarray, list, int], None]] = None
    ) -> int:
        """Detect objects in a video file.

        Args:
            video_path: Path to input video file
            output_dir: Optional directory to save frame images (frame_{n}.png)
            display: Whether to display results in window
            progress_bar: Whether to show tqdm progress bar
            callback: Optional callback function(frame, boxes, frame_number)

        Returns:
            Total number of frames processed
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f'Failed to open video file: {video_path}')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.verbose:
            print(f'Processing video: {video_path} ({total_frames} frames)')

        # Create output directory if needed
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        frame_number = 0
        pbar = tqdm(total=total_frames, desc='Processing video') if progress_bar else None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect objects
                result_img, boxes = self.detect_and_visualize_numpy(frame, save_path=None)

                # Save frame if output directory specified
                if output_dir is not None:
                    output_path = str(Path(output_dir) / f'{str(frame_number).zfill(5)}.png')
                    cv2.imwrite(output_path, result_img)

                # Call user callback if provided
                if callback is not None:
                    callback(result_img, boxes, frame_number)

                # Display results
                if display:
                    cv2.imshow('Result', result_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_number += 1
                if pbar is not None:
                    pbar.update(1)

        finally:
            if pbar is not None:
                pbar.close()
            cap.release()
            if display:
                cv2.destroyAllWindows()

        if self.verbose:
            print(f'Processed {frame_number} frames')

        return frame_number
