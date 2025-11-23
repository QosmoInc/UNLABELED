"""Single image detection script.

Refactored version of detect.py using ImageDetector class.
"""

import sys
from evaluation.detectors import ImageDetector


def main() -> None:
    """Run detection on a single image."""
    if len(sys.argv) != 4:
        print('Usage:')
        print('  python -m evaluation.detect_single_image cfgfile weightfile imgfile')
        sys.exit(1)

    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    imgfile = sys.argv[3]

    # Create detector
    detector = ImageDetector(
        cfgfile=cfgfile,
        weightfile=weightfile,
        conf_thresh=0.5,
        nms_thresh=0.4,
        use_cuda=True,
        verbose=True
    )

    # Detect objects in image
    boxes = detector.detect_image_file(imgfile, save_path=imgfile)

    print(f'Detection complete. Found {len(boxes)} objects.')
    print(f'Results saved to: {imgfile}')


if __name__ == '__main__':
    main()
