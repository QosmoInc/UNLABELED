"""Base trainer class for adversarial patch training.

This module provides an abstract base class that contains common functionality
shared across all patch trainer implementations.
"""

import os
import random
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import torch
import torchvision.transforms as transforms
from PIL import Image

from darknet import Darknet
from load_data import (
    PatchTransformer,
    PatchApplier,
    MaxProbExtractor,
    AdaINStyleLoss,
    ContentLoss,
    TotalVariation
)
from experiment_tracker import ExperimentTracker
from logger import setup_logger

if TYPE_CHECKING:
    from config_models import TrainingConfig


class BasePatchTrainer(ABC):
    """Abstract base class for adversarial patch trainers.

    This class provides common functionality for all patch training scenarios:
    - Model initialization (YOLO detection, loss functions)
    - Patch generation and I/O operations
    - Experiment tracking with WandB
    - Device management
    - Reproducible training with seed setting

    Subclasses must implement the train() method with dataset-specific logic.
    """

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # Set deterministic behavior for CUDA operations
        # Note: This may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __init__(self, config: 'TrainingConfig', device: Optional[str] = None) -> None:
        """Initialize the patch trainer.

        Args:
            config: TrainingConfig instance with all configuration parameters
            device: Device to use ('cuda:0', 'cpu', etc.). Overrides config.trainer.device if specified.

        Raises:
            FileNotFoundError: If required config files don't exist
        """
        # Store configuration
        self.config: 'TrainingConfig' = config

        # Setup logger
        self.logger = setup_logger(
            name=self.__class__.__name__,
            level=config.logging.level,
            log_dir=Path(config.logging.log_dir) if config.logging.log_dir else None,
            console=config.logging.console,
            file=config.logging.file,
        )

        # Set random seed for reproducibility if specified
        if self.config.training.seed is not None:
            self.logger.info(f'Setting random seed to: {self.config.training.seed}')
            self.set_seed(self.config.training.seed)
            self.logger.info('Deterministic mode enabled for reproducibility')
        else:
            self.logger.info('No seed specified - training will be non-deterministic')

        # Device setup with fallback
        # Priority: explicit device arg > config.trainer.device > auto-detect
        if device is None:
            device = config.trainer.device

        if device is None:
            if torch.cuda.is_available():
                self.device: str = 'cuda:0'
                self.logger.info('CUDA available - using GPU')
            else:
                self.device = 'cpu'
                self.logger.warning('CUDA not available - falling back to CPU (training will be slower)')
        else:
            self.device = device
            if 'cuda' in device and not torch.cuda.is_available():
                self.logger.warning(f'Requested device "{device}" but CUDA not available')
                self.logger.warning('Falling back to CPU')
                self.device = 'cpu'

        self.logger.info(f'Device: {self.device}')
        self.logger.info('=' * 40)

        # Initialize YOLO detection model
        self.darknet_model: Darknet = Darknet(self.config.model.cfgfile)
        self.darknet_model.load_weights(self.config.model.weightfile)
        # Set to eval mode to disable dropout/batch norm training behavior
        self.darknet_model = self.darknet_model.eval().to(self.device)

        # Initialize patch transformation modules
        self.patch_applier: PatchApplier = PatchApplier().to(self.device)
        self.patch_transformer: PatchTransformer = PatchTransformer().to(self.device)

        # Initialize experiment tracker (WandB)
        self.tracker: ExperimentTracker = ExperimentTracker(
            config=self.config,
            experiment_name=self.config.patch.name,
            enable_wandb=self.config.wandb.enabled,
            wandb_project=self.config.wandb.project,
            wandb_entity=self.config.wandb.entity,
            wandb_tags=self.config.wandb.tags,
            wandb_notes=self.config.wandb.notes
        )

        # Track epoch length for logging
        self.epoch_length: int = 0

    def generate_patch(self, patch_type: str = 'gray') -> torch.Tensor:
        """Generate an initial patch as starting point for optimization.

        Args:
            patch_type: Type of patch to generate
                - 'gray': Uniform gray patch (0.5 in all channels)
                - 'random': Random noise patch

        Returns:
            torch.Tensor: Patch tensor of shape (3, patch_size, patch_size)
        """
        if patch_type == 'gray':
            adv_patch_cpu = torch.full(
                (3, self.config.patch.size, self.config.patch.size),
                0.5
            )
        elif patch_type == 'random':
            adv_patch_cpu = torch.rand(
                (3, self.config.patch.size, self.config.patch.size)
            )
        else:
            raise ValueError(f"Unknown patch type: {patch_type}. Use 'gray' or 'random'.")

        return adv_patch_cpu

    def read_image(self, path: str) -> torch.Tensor:
        """Read an image file and convert it to a patch tensor.

        Args:
            path: Path to the image file

        Returns:
            torch.Tensor: Image tensor resized to (3, patch_size, patch_size)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image file is invalid or corrupted
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        try:
            patch_img = Image.open(path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image {path}: {str(e)}")

        # Resize to patch size
        tf = transforms.Resize((self.config.patch.size, self.config.patch.size))
        patch_img = tf(patch_img)

        # Convert to tensor
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)

        return adv_patch_cpu

    def save_patch(self, adv_patch_cpu: torch.Tensor, epoch: int, ep_det_loss: Optional[float] = None) -> Any:
        """Save the current patch to disk and track as artifact.

        Args:
            adv_patch_cpu: Patch tensor to save
            epoch: Current epoch number (used in filename)
            ep_det_loss: Optional detection loss value (included in filename)

        Returns:
            Path to saved patch file
        """
        if ep_det_loss is None:
            ep_det_loss = 0.0

        # Save using experiment tracker
        return self.tracker.save_patch_artifact(
            patch=adv_patch_cpu,
            epoch=epoch,
            loss=ep_det_loss,
            is_best=False
        )

    def save_best_patch(self, adv_patch_cpu: torch.Tensor, epoch: int, det_loss: float) -> None:
        """Save the best performing patch so far and track as artifact.

        Args:
            adv_patch_cpu: Patch tensor to save
            epoch: Current epoch number
            det_loss: Detection loss value
        """
        # Save using experiment tracker
        self.tracker.save_patch_artifact(
            patch=adv_patch_cpu,
            epoch=epoch,
            loss=det_loss,
            is_best=True
        )

    def cleanup_memory(self, *tensors: Any) -> None:
        """Clean up GPU memory by deleting tensors and clearing cache.

        Args:
            *tensors: Variable number of tensors to delete
        """
        for tensor in tensors:
            del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def train(self) -> None:
        """Train the adversarial patch.

        This method must be implemented by subclasses with dataset-specific
        training logic.
        """
        raise NotImplementedError("Subclasses must implement train() method")
