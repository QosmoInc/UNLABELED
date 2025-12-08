"""Video file detection script.

Refactored version of detect_video.py using VideoDetector class.
"""

import argparse
from pathlib import Path
from evaluation.detectors import VideoDetector


def main() -> None:
    """Run detection on a video file."""
    parser = argparse.ArgumentParser(
        description='Run object detection on a video file (Press "q" to quit during playback)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n'
               '  python -m evaluation.detect_video --cfgfile cfg/yolo.cfg --weightfile weights/yolo.weights --video-path video.mp4\n'
               '  python -m evaluation.detect_video --cfgfile cfg/yolo.cfg --weightfile weights/yolo.weights --video-path video.mp4 --output-dir output_frames/'
    )
    parser.add_argument('--cfgfile', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--weightfile', type=str, required=True, help='Path to model weights file')
    parser.add_argument('--video-path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save frame images (default: None)')
    parser.add_argument('--conf-thresh', type=float, default=0.75, help='Confidence threshold (default: 0.75)')
    parser.add_argument('--nms-thresh', type=float, default=0.4, help='NMS threshold (default: 0.4)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA (use CPU)')
    parser.add_argument('--quiet', action='store_true', help='Disable verbose output')
    parser.add_argument('--display', action='store_true', help='Enable display window during processing')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')

    args = parser.parse_args()

    # Validate video file exists
    if not Path(args.video_path).exists():
        print(f'Error: Video file not found: {args.video_path}')
        return

    # Create detector
    detector = VideoDetector(
        cfgfile=args.cfgfile,
        weightfile=args.weightfile,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        use_cuda=not args.no_cuda,
        verbose=not args.quiet
    )

    # Run video detection
    num_frames = detector.detect_video_file(
        video_path=args.video_path,
        output_dir=args.output_dir,
        display=args.display,
        progress_bar=not args.no_progress
    )

    print(f'\nVideo detection complete!')
    print(f'Processed {num_frames} frames')
    if args.output_dir is not None:
        print(f'Frames saved to: {args.output_dir}/')


if __name__ == '__main__':
    main()
