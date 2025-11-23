"""Animal-specific patch trainers (Cat and Dog).

This module implements trainers for cat and dog detection using INRIA dataset
with different target class IDs from the COCO dataset.

COCO class IDs:
- 15: cat
- 16: dog
"""

from typing import Optional

from .inria_trainer import InriaPatchTrainer
from load_data import MaxProbExtractor


class CatPatchTrainer(InriaPatchTrainer):
    """Patch trainer targeting cat detection (COCO class 15).

    This trainer generates adversarial patches that reduce cat detection
    confidence in object detection models.
    """

    def __init__(self, mode: str, device: Optional[str] = None) -> None:
        """Initialize Cat trainer.

        Args:
            mode: Configuration name from patch_config.patch_configs
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
        """
        super().__init__(mode, device)

        # Override prob_extractor to target cat class (ID: 15)
        # COCO dataset has 80 classes total
        self.prob_extractor: MaxProbExtractor = MaxProbExtractor(
            15, 80, self.config
        ).to(self.device)


class DogPatchTrainer(InriaPatchTrainer):
    """Patch trainer targeting dog detection (COCO class 16).

    This trainer generates adversarial patches that reduce dog detection
    confidence in object detection models.
    """

    def __init__(self, mode: str, device: Optional[str] = None) -> None:
        """Initialize Dog trainer.

        Args:
            mode: Configuration name from patch_config.patch_configs
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
        """
        super().__init__(mode, device)

        # Override prob_extractor to target dog class (ID: 16)
        # COCO dataset has 80 classes total
        self.prob_extractor: MaxProbExtractor = MaxProbExtractor(
            16, 80, self.config
        ).to(self.device)
