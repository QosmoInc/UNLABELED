"""
Training code for Adversarial Patch Training on INRIA Person Dataset

This script trains adversarial patches to fool YOLOv2 person detection models.
Based on the paper: "Fooling automated surveillance cameras: adversarial patches
to attack person detection" (CVPRW 2019).

Main training script using:
- INRIA Person Dataset for realistic person images
- YOLOv2 MS COCO weights for detection
- Various loss functions: detection loss, style loss, content loss, TV loss

Usage:
    python train_patch_inria.py <config_name>

    where <config_name> is a key from patch_config.patch_configs
    Example: python train_patch_inria.py paper_obj
"""

import sys
import patch_config
from trainers import InriaPatchTrainer


def main():
    """Main entry point for training."""
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)
        sys.exit(1)

    # Create trainer instance and start training
    trainer = InriaPatchTrainer(sys.argv[1])
    trainer.train()


if __name__ == '__main__':
    main()
