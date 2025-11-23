"""Load data package for adversarial patch training.

This package provides modular components for adversarial patch generation:
- losses: Loss function modules (MaxProbExtractor, NPSCalculator, TotalVariation, ContentLoss, AdaINStyleLoss)
- transforms: Patch transformation modules (PatchTransformer, PatchApplier, PatchGenerator)
- dataset: Dataset loader (InriaDataset)

For backward compatibility, all classes are imported at the package level.
This allows existing code to continue using:
    from load_data import MaxProbExtractor, PatchTransformer, InriaDataset, ...
"""

# Import all classes for backward compatibility
from .losses import (
    MaxProbExtractor,
    NPSCalculator,
    TotalVariation,
    ContentLoss,
    AdaINStyleLoss,
)

from .transforms import (
    PatchTransformer,
    PatchApplier,
    PatchGenerator,
)

from .dataset import (
    InriaDataset,
)

# Define what's exported when using "from load_data import *"
__all__ = [
    # Loss functions
    'MaxProbExtractor',
    'NPSCalculator',
    'TotalVariation',
    'ContentLoss',
    'AdaINStyleLoss',
    # Transforms
    'PatchTransformer',
    'PatchApplier',
    'PatchGenerator',
    # Dataset
    'InriaDataset',
]
