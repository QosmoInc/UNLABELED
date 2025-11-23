"""Video file detection script.

Refactored version of detect_video.py using VideoDetector class.
"""

import sys
from pathlib import Path
from evaluation.detectors import VideoDetector


def main() -> None:
    """Run detection on a video file."""
    if len(sys.argv) not in [3, 4]:
        print('Usage:')
        print('  python -m evaluation.detect_video cfgfile weightfile video_path [output_dir]')
        print('')
        print('Arguments:')
        print('  video_path: Path to input video file')
        print('  output_dir: Optional directory to save frame images (default: None)')
        print('')
        print('Example:')
        print('  python -m evaluation.detect_video cfg/yolo.cfg weights/yolo.weights video.mp4')
        print('  python -m evaluation.detect_video cfg/yolo.cfg weights/yolo.weights video.mp4 output_frames/')
        print('')
        print('Press "q" to quit during playback')
        sys.exit(1)

    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    video_path = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) == 4 else None

    # Validate video file exists
    if not Path(video_path).exists():
        print(f'Error: Video file not found: {video_path}')
        sys.exit(1)

    # Create detector
    detector = VideoDetector(
        cfgfile=cfgfile,
        weightfile=weightfile,
        conf_thresh=0.75,  # Higher threshold for video
        nms_thresh=0.4,
        use_cuda=True,
        verbose=True
    )

    # Run video detection
    num_frames = detector.detect_video_file(
        video_path=video_path,
        output_dir=output_dir,
        display=False,
        progress_bar=True
    )

    print(f'\nVideo detection complete!')
    print(f'Processed {num_frames} frames')
    if output_dir is not None:
        print(f'Frames saved to: {output_dir}/')


if __name__ == '__main__':
    main()
