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
`patch_config.py` contains configuration of different experiments. You can design your own experiment by inheriting from the base `BaseConfig` class or an existing experiment. `ReproducePaperObj` reproduces the patch that minimizes object score from the paper (With a lower batch size to fit on a desktop GPU).

You can generate this patch by running:
```bash
uv run python train_patch_inria.py paper_obj
```

Note: `train_patch_inria.py` is the main training script for person detection on the INRIA dataset.
