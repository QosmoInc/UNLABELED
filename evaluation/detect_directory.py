"""Directory/batch image detection script.

Refactored version of detect_dir.py using ImageDetector class.
"""

import argparse
from evaluation.detectors import ImageDetector


def main() -> None:
    """Run detection on all images in a directory."""
    parser = argparse.ArgumentParser(
        description='Run object detection on all images in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n'
               '  python -m evaluation.detect_directory --cfgfile cfg/yolo.cfg --weightfile weights/yolo.weights --img-dir data/images/'
    )
    parser.add_argument('--cfgfile', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--weightfile', type=str, required=True, help='Path to model weights file')
    parser.add_argument('--img-dir', type=str, required=True, help='Path to directory containing images')
    parser.add_argument('--pattern', type=str, default='*.png', help='File pattern to match (default: *.png)')
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

    # Detect objects in all images
    results = detector.detect_directory(args.img_dir, pattern=args.pattern)

    print(f'\nBatch detection complete!')
    print(f'Processed {len(results)} images')
    print(f'Results saved to original image paths')


if __name__ == '__main__':
    main()
