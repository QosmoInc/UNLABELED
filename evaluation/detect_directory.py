"""Directory/batch image detection script.

Refactored version of detect_dir.py using ImageDetector class.
"""

import sys
from evaluation.detectors import ImageDetector


def main() -> None:
    """Run detection on all images in a directory."""
    if len(sys.argv) != 4:
        print('Usage:')
        print('  python -m evaluation.detect_directory cfgfile weightfile img_dir')
        print('')
        print('Example:')
        print('  python -m evaluation.detect_directory cfg/yolo.cfg weights/yolo.weights data/images/')
        sys.exit(1)

    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    img_dir = sys.argv[3]

    # Create detector
    detector = ImageDetector(
        cfgfile=cfgfile,
        weightfile=weightfile,
        conf_thresh=0.5,
        nms_thresh=0.4,
        use_cuda=True,
        verbose=True
    )

    # Detect objects in all images
    results = detector.detect_directory(img_dir, pattern='*.png')

    print(f'\nBatch detection complete!')
    print(f'Processed {len(results)} images')
    print(f'Results saved to original image paths')


if __name__ == '__main__':
    main()
