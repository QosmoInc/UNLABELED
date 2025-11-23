"""Training script for adversarial patch with Unity synthetic dataset.

This script provides a command-line interface to train adversarial patches
using the Unity-generated synthetic dataset.

Usage:
    python train_patch.py <config_mode>

Example:
    python train_patch.py exp1
"""

import sys
import patch_config
from trainers import UnityPatchTrainer


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)
        sys.exit(1)

    trainer = UnityPatchTrainer(sys.argv[1])
    trainer.train()


if __name__ == '__main__':
    main()
