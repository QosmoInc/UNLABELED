"""Single image detection script.

Refactored version of detect.py using ImageDetector class.
"""

import argparse
from evaluation.detectors import ImageDetector


def main() -> None:
    """Run detection on a single image."""
    parser = argparse.ArgumentParser(
        description='Run object detection on a single image',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--cfgfile', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--weightfile', type=str, required=True, help='Path to model weights file')
    parser.add_argument('--imgfile', type=str, required=True, help='Path to input image file')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--nms-thresh', type=float, default=0.4, help='NMS threshold (default: 0.4)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA (use CPU)')
    parser.add_argument('--quiet', action='store_true', help='Disable verbose output')

    args = parser.parse_args()

    # Create detector
    detector = ImageDetector(
        cfgfile=args.cfgfile,
        weightfile=args.weightfile,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        use_cuda=not args.no_cuda,
        verbose=not args.quiet
    )

    # Detect objects in image
    boxes = detector.detect_image_file(args.imgfile, save_path=args.imgfile)

    print(f'Detection complete. Found {len(boxes)} objects.')
    print(f'Results saved to: {args.imgfile}')


if __name__ == '__main__':
    main()
