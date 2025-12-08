"""Camera/webcam detection script.

Refactored version of detect_cam.py using VideoDetector class.
"""

import argparse
from evaluation.detectors import VideoDetector


def main() -> None:
    """Run detection on camera/webcam stream."""
    parser = argparse.ArgumentParser(
        description='Run object detection on camera/webcam stream (Press "q" to quit)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n'
               '  python -m evaluation.detect_camera --cfgfile cfg/yolo.cfg --weightfile weights/yolo.weights\n'
               '  python -m evaluation.detect_camera --cfgfile cfg/yolo.cfg --weightfile weights/yolo.weights --camera-id 1'
    )
    parser.add_argument('--cfgfile', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--weightfile', type=str, required=True, help='Path to model weights file')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--nms-thresh', type=float, default=0.4, help='NMS threshold (default: 0.4)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA (use CPU)')
    parser.add_argument('--quiet', action='store_true', help='Disable verbose output')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    parser.add_argument('--no-fullscreen', action='store_true', help='Disable fullscreen mode')
    parser.add_argument('--filter-person', action='store_true', help='Filter detections to show only person class')

    args = parser.parse_args()

    # Set filter_classes based on --filter-person flag
    # Person class ID is 0 in COCO dataset
    filter_classes = [0] if args.filter_person else None

    # Create detector
    detector = VideoDetector(
        cfgfile=args.cfgfile,
        weightfile=args.weightfile,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        use_cuda=not args.no_cuda,
        verbose=not args.quiet,
        filter_classes=filter_classes
    )

    # Run camera detection
    detector.detect_camera(
        camera_id=args.camera_id,
        display=not args.no_display,
        fullscreen=not args.no_fullscreen
    )

    print('Camera detection stopped.')


if __name__ == '__main__':
    main()
