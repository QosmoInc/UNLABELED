# Evaluation Tools

This directory contains refactored evaluation and detection scripts with a clean, modular architecture.

## 📁 Structure

```
evaluation/
├── detectors/              # Object detection modules
│   ├── base_detector.py    # Base class for all detectors
│   ├── image_detector.py   # Single/batch image detection
│   └── video_detector.py   # Video/camera detection
├── patch_evaluator.py      # Adversarial patch evaluation
├── check_recognition_rate.py  # Patch effectiveness measurement
├── detect_single_image.py  # CLI: Single image detection
├── detect_directory.py     # CLI: Batch image detection
├── detect_camera.py        # CLI: Camera/webcam detection
├── detect_video.py         # CLI: Video file detection
└── __init__.py             # Package exports
```

## 🚀 Usage

### Object Detection

#### Single Image
```bash
python -m evaluation.detect_single_image cfg/yolo.cfg weights/yolo.weights image.jpg
```

#### Directory/Batch
```bash
python -m evaluation.detect_directory cfg/yolo.cfg weights/yolo.weights data/images/
```

#### Camera/Webcam
```bash
# Default camera (ID: 0)
python -m evaluation.detect_camera cfg/yolo.cfg weights/yolo.weights

# Specific camera
python -m evaluation.detect_camera cfg/yolo.cfg weights/yolo.weights 1
```

Press `q` to quit camera view.

#### Video File
```bash
# Process video without saving frames
python -m evaluation.detect_video cfg/yolo.cfg weights/yolo.weights video.mp4

# Process video and save detected frames
python -m evaluation.detect_video cfg/yolo.cfg weights/yolo.weights video.mp4 output_frames/
```

### Patch Evaluation

#### Check Recognition Rate
```bash
# Basic usage
python -m evaluation.check_recognition_rate paper_obj path/to/patch.png

# With custom parameters
python -m evaluation.check_recognition_rate paper_obj patch.png \
    --target-class 0 \
    --patch-size 300 \
    --batch-size 8 \
    --num-batches 50 \
    --no-rotate

# Help
python -m evaluation.check_recognition_rate --help
```

## 💻 Programmatic Usage

### Image Detection
```python
from evaluation import ImageDetector

# Create detector
detector = ImageDetector(
    cfgfile='cfg/yolo.cfg',
    weightfile='weights/yolo.weights',
    conf_thresh=0.5,
    nms_thresh=0.4,
    use_cuda=True
)

# Detect single image
boxes = detector.detect_image_file('image.jpg')

# Detect directory
results = detector.detect_directory('data/images/', pattern='*.png')
```

### Video Detection
```python
from evaluation import VideoDetector

# Create detector
detector = VideoDetector(
    cfgfile='cfg/yolo.cfg',
    weightfile='weights/yolo.weights'
)

# Camera stream
detector.detect_camera(camera_id=0, display=True)

# Video file
num_frames = detector.detect_video_file(
    video_path='video.mp4',
    output_dir='output_frames/',
    progress_bar=True
)
```

### Patch Evaluation
```python
from evaluation import PatchEvaluator

# Create evaluator
evaluator = PatchEvaluator(
    mode='paper_obj',
    target_class=0,  # person
    num_classes=80   # COCO
)

# Evaluate patch effectiveness
results = evaluator.evaluate_patch_on_dataset(
    patch_path='patch.png',
    patch_size=300,
    batch_size=8,
    num_batches=10
)

print(f"Mean detection probability: {results['mean_prob']:.4f}")

# Compare with/without patch
without_patch, with_patch = evaluator.compare_with_without_patch(
    patch_path='patch.png',
    num_batches=10
)
```

## 🔧 Key Improvements

### Eliminated Code Duplication
- **Before**: 5 detect scripts with ~150 lines each = 750+ lines of duplicated code
- **After**: 3 base classes + 4 thin CLI wrappers = ~600 lines total
- **Reduction**: ~150 lines (20% reduction) with better organization

### Type Safety
- All classes and functions have comprehensive type hints
- Better IDE support and error detection

### Consistent Interface
- All detection scripts follow the same pattern
- Easy to extend with new detection modes

### Better Modularity
- Core logic in base classes
- CLI scripts are thin wrappers
- Easy to test and maintain

## 📝 Migration from Old Scripts

| Old Script | New Script | Notes |
|------------|-----------|-------|
| `detect.py` | `evaluation.detect_single_image` | Same functionality |
| `detect_image.py` | `evaluation.detect_single_image` | Merged with detect.py |
| `detect_dir.py` | `evaluation.detect_directory` | Same functionality |
| `detect_cam.py` | `evaluation.detect_camera` | Same functionality |
| `detect_video.py` | `evaluation.detect_video` | Same functionality |
| `check_recognition_rate.py` | `evaluation.check_recognition_rate` | Enhanced with argparse |
| `test_patch_inria.py` | Use `PatchEvaluator` class | Better interface |

Old scripts are archived in `archive/old_scripts/` for reference.

## 🎯 Architecture

### BaseDetector
Base class providing common functionality:
- Model loading and initialization
- CUDA device management
- Class name loading (COCO/VOC)
- Image preprocessing
- Detection and visualization

### ImageDetector
Extends BaseDetector for image processing:
- Single image detection
- Batch/directory processing
- Flexible output paths

### VideoDetector
Extends BaseDetector for video/camera:
- Camera stream processing
- Video file processing
- Frame-by-frame callbacks
- Progress tracking

### PatchEvaluator
Specialized for adversarial patch evaluation:
- Patch loading and preprocessing
- Dataset-based evaluation
- Statistical analysis
- Comparison metrics
