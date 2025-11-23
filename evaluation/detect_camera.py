"""Camera/webcam detection script.

Refactored version of detect_cam.py using VideoDetector class.
"""

import sys
from evaluation.detectors import VideoDetector


def main() -> None:
    """Run detection on camera/webcam stream."""
    if len(sys.argv) not in [3, 4]:
        print('Usage:')
        print('  python -m evaluation.detect_camera cfgfile weightfile [camera_id]')
        print('')
        print('Arguments:')
        print('  camera_id: Camera device ID (default: 0)')
        print('')
        print('Example:')
        print('  python -m evaluation.detect_camera cfg/yolo.cfg weights/yolo.weights')
        print('  python -m evaluation.detect_camera cfg/yolo.cfg weights/yolo.weights 1')
        print('')
        print('Press "q" to quit')
        sys.exit(1)

    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    camera_id = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    # Create detector
    detector = VideoDetector(
        cfgfile=cfgfile,
        weightfile=weightfile,
        conf_thresh=0.5,
        nms_thresh=0.4,
        use_cuda=True,
        verbose=True
    )

    # Run camera detection
    detector.detect_camera(
        camera_id=camera_id,
        display=True,
        fullscreen=True
    )

    print('Camera detection stopped.')


if __name__ == '__main__':
    main()
