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

## Method 1: Unified Training Script (Recommended)

The project now includes a unified training script that supports multiple training modes through YAML configuration files.

### List available trainers
```bash
uv run python train_patch_unified.py --list-trainers
```

### Train with a configuration file
```bash
# INRIA person detection (reproduces paper results)
uv run python train_patch_unified.py --config configs/person_inria.yaml

# Cat detection
uv run python train_patch_unified.py --config configs/cat_inria.yaml

# Dog detection (Unity dataset)
uv run python train_patch_unified.py --config configs/dog_unity.yaml

# Multi-class: suppress person detection while enhancing bear detection
uv run python train_patch_unified.py --config configs/person_bear_multiclass.yaml
```

### Override configuration parameters
```bash
uv run python train_patch_unified.py --config configs/person_inria.yaml \
    --batch-size 8 \
    --epochs 500 \
    --learning-rate 0.01
```

### Available Trainers
- **InriaPatchTrainer**: Person detection using INRIA dataset
- **CatPatchTrainer**: Cat detection (COCO class 15)
- **DogPatchTrainer**: Dog detection (COCO class 16)
- **MultiClassPatchTrainer**: Multi-class adversarial patches (suppress one class, enhance another)
- **UnityPatchTrainer**: Unity-generated synthetic data (requires `python-osc`)

### Configuration Files
Configuration files are located in the `configs/` directory:
- `person_inria.yaml` - INRIA person detection (paper reproduction)
- `cat_inria.yaml` - Cat detection
- `dog_unity.yaml` - Dog detection with Unity dataset
- `person_bear_multiclass.yaml` - Multi-class training
- `unity_person.yaml` - Unity person detection

You can create your own configuration files based on these examples.

## Method 2: Direct Training Scripts (Legacy)

`patch_config.py` contains configuration of different experiments. You can design your own experiment by inheriting from the base `BaseConfig` class or an existing experiment. `ReproducePaperObj` reproduces the patch that minimizes object score from the paper (With a lower batch size to fit on a desktop GPU).

You can generate this patch by running:
```bash
uv run python train_patch_inria.py paper_obj
```

Note: `train_patch_inria.py` is the main training script for person detection on the INRIA dataset.

### Other Training Scripts
- `train_patch.py` - Unity synthetic data
- `archive/training_variants/train_patch_dog.py` - Dog detection
- `archive/training_variants/train_patch_inria_cat.py` - Cat detection
- `archive/training_variants/train_patch_inria_class_up.py` - Multi-class training
