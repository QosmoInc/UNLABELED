"""Trainers package for adversarial patch training.

This package provides modular trainer classes for different training scenarios:
- BasePatchTrainer: Abstract base class with common functionality
- InriaPatchTrainer: Trainer for INRIA Person Dataset
- UnityPatchTrainer: Trainer for Unity-generated synthetic data
- CatPatchTrainer: Trainer targeting cat detection
- DogPatchTrainer: Trainer targeting dog detection
- MultiClassPatchTrainer: Trainer for multi-class adversarial patches

For backward compatibility, trainers can be imported at the package level:
    from trainers import InriaPatchTrainer, UnityPatchTrainer, CatPatchTrainer
"""

from .base_trainer import BasePatchTrainer
from .inria_trainer import InriaPatchTrainer
from .animal_trainers import CatPatchTrainer, DogPatchTrainer
from .multiclass_trainer import MultiClassPatchTrainer

# Optional imports (may require additional dependencies)
try:
    from .unity_trainer import UnityPatchTrainer
    _UNITY_AVAILABLE = True
except ImportError:
    _UNITY_AVAILABLE = False
    UnityPatchTrainer = None  # type: ignore

__all__ = [
    'BasePatchTrainer',
    'InriaPatchTrainer',
    'CatPatchTrainer',
    'DogPatchTrainer',
    'MultiClassPatchTrainer',
]

if _UNITY_AVAILABLE:
    __all__.append('UnityPatchTrainer')
