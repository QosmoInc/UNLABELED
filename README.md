- - -
This repository is the fork of https://gitlab.com/EAVISE/adversarial-yolo
- - -

# Adversarial YOLO
This repository is based on the marvis YOLOv2 inplementation: https://github.com/marvis/pytorch-yolo2

This work corresponds to the following paper: https://arxiv.org/abs/1904.08653:
```
@inproceedings{thysvanranst2019,
    title={Fooling automated surveillance cameras: adversarial patches to attack person detection},
    author={Thys, Simen and Van Ranst, Wiebe and Goedem\'e, Toon},
    booktitle={CVPRW: Workshop on The Bright and Dark Sides of Computer Vision: Challenges and Opportunities for Privacy and Security},
    year={2019}
}
```

If you use this work, please cite this paper.

# What you need
This project uses Python 3.11 and the `uv` package manager.

## Installation

1. Install Python 3.11 (if not already installed)
2. Install `uv` package manager: https://github.com/astral-sh/uv
3. Install dependencies:
```bash
uv sync
```

This will install PyTorch 2.9.1 (with CUDA 12.9), TensorBoard, and other dependencies as specified in `pyproject.toml`.

Make sure you have the YOLOv2 MS COCO weights:
```
mkdir weights; curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolo.weights
```

Get the INRIA dataset:
```
curl ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -o inria.tar
tar xf inria.tar
mv INRIAPerson inria
cp -r yolo-labels inria/Train/pos/
```

# Generating a patch

## Training with YAML Configuration (Recommended)

The project uses a type-safe YAML-based configuration system with Pydantic validation. All training is done through the unified training script.

### Quick Start

1. **Validate configuration** (checks files exist and parameters are valid):
```bash
uv run python train_patch_unified.py --config configs/person_inria.yaml --validate-only
```

2. **Start training**:
```bash
uv run python train_patch_unified.py --config configs/person_inria.yaml
```

### Available Configuration Files

All configuration files are located in the `configs/` directory with Pydantic validation:

| Config File | Trainer | Dataset | Target | Description |
|------------|---------|---------|--------|-------------|
| `person_inria.yaml` | InriaPatchTrainer | INRIA | Person (class 0) | Paper reproduction |
| `cat_inria.yaml` | CatPatchTrainer | INRIA | Cat (class 15) | Cat detection |
| `dog_unity.yaml` | DogPatchTrainer | Unity | Dog (class 16) | Dog detection |
| `person_bear_multiclass.yaml` | MultiClassPatchTrainer | INRIA | Person↓ Bear↑ | Multi-class |
| `unity_person.yaml` | UnityPatchTrainer | Unity | Person (class 0) | Unity synthetic |

### Training Examples

```bash
# INRIA person detection (reproduces paper results)
uv run python train_patch_unified.py --config configs/person_inria.yaml

# Cat detection
uv run python train_patch_unified.py --config configs/cat_inria.yaml

# Dog detection (Unity dataset)
uv run python train_patch_unified.py --config configs/dog_unity.yaml

# Multi-class: suppress person while enhancing bear
uv run python train_patch_unified.py --config configs/person_bear_multiclass.yaml
```

### Override Configuration Parameters

You can override any parameter via command-line arguments:

```bash
uv run python train_patch_unified.py --config configs/person_inria.yaml \
    --batch-size 16 \
    --epochs 500 \
    --learning-rate 0.01 \
    --patch-size 400
```

### List Available Trainers

```bash
uv run python train_patch_unified.py --list-trainers
```

Output:
- **InriaPatchTrainer**: Person detection using INRIA dataset
- **CatPatchTrainer**: Cat detection (COCO class 15)
- **DogPatchTrainer**: Dog detection (COCO class 16)
- **MultiClassPatchTrainer**: Suppress one class, enhance another
- **UnityPatchTrainer**: Unity synthetic data (requires `python-osc`)

### Creating Custom Configurations

Create a new YAML file in `configs/` directory. See existing files for examples:

```yaml
# configs/my_custom.yaml
trainer:
  type: InriaPatchTrainer

dataset:
  type: inria
  img_dir: inria/Train/pos
  lab_dir: inria/Train/pos/yolo-labels
  max_labels: 14

model:
  cfgfile: cfg/yolov2.cfg
  weightfile: weights/yolov2.weights

patch:
  size: 300
  name: my_custom_patch
  initial_type: gray  # Options: gray, random, image

training:
  batch_size: 8
  epochs: 300
  learning_rate: 0.03
  num_workers: 4

losses:
  detection_weight: 1.0
  tv_weight: 0.5
  tv_max: 0.165

target:
  class_id: 0  # COCO class ID
  objective: minimize  # or maximize
```

Then validate and run:
```bash
uv run python train_patch_unified.py --config configs/my_custom.yaml --validate-only
uv run python train_patch_unified.py --config configs/my_custom.yaml
```

## Legacy Training Scripts (Deprecated)

**Note**: The old `patch_config.py` system has been replaced with YAML configuration. Legacy scripts are kept in `archive/` for reference but are no longer maintained.

If you need to use legacy scripts:
```bash
# Legacy training (not recommended)
uv run python archive/training_variants/train_patch_dog.py
```

For new projects, use the YAML configuration system above
