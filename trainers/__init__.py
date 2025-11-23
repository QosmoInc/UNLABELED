"""Trainers package for adversarial patch training.

This package provides modular trainer classes for different training scenarios:
- BasePatchTrainer: Abstract base class with common functionality
- InriaPatchTrainer: Trainer for INRIA Person Dataset
- UnityPatchTrainer: Trainer for Unity-generated synthetic data

For backward compatibility, trainers can be imported at the package level:
    from trainers import InriaPatchTrainer, UnityPatchTrainer
"""

from .base_trainer import BasePatchTrainer
from .inria_trainer import InriaPatchTrainer
from .unity_trainer import UnityPatchTrainer

__all__ = [
    'BasePatchTrainer',
    'InriaPatchTrainer',
    'UnityPatchTrainer',
]
