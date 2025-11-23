"""Base trainer class for adversarial patch training.

This module provides an abstract base class that contains common functionality
shared across all patch trainer implementations.
"""

import os
import time
import subprocess
from abc import ABC, abstractmethod

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
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
import patch_config


class BasePatchTrainer(ABC):
    """Abstract base class for adversarial patch trainers.

    This class provides common functionality for all patch training scenarios:
    - Model initialization (YOLO detection, loss functions)
    - Patch generation and I/O operations
    - TensorBoard logging setup
    - Device management

    Subclasses must implement the train() method with dataset-specific logic.
    """

    def __init__(self, mode: str, device: str = None):
        """Initialize the patch trainer.

        Args:
            mode: Configuration name from patch_config.patch_configs
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
        """
        # Load configuration
        self.config = patch_config.patch_configs[mode]()
        self.mode = mode

        # Device setup
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f'Device: {self.device}')
        print(self.config)
        print('=' * 40)

        # Initialize YOLO detection model
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        # Set to eval mode to disable dropout/batch norm training behavior
        self.darknet_model = self.darknet_model.eval().to(self.device)

        # Initialize patch transformation modules
        self.patch_applier = PatchApplier().to(self.device)
        self.patch_transformer = PatchTransformer().to(self.device)

        # Initialize TensorBoard writer
        self.writer = self.init_tensorboard(mode)

        # Track epoch length for logging
        self.epoch_length = 0

    def init_tensorboard(self, name: str = None):
        """Initialize TensorBoard logging.

        Args:
            name: Optional name for the experiment. Timestamped if provided.

        Returns:
            SummaryWriter instance for logging
        """
        # Launch TensorBoard server in background
        subprocess.Popen(
            ['tensorboard', '--logdir=runs'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def generate_patch(self, patch_type: str = 'gray'):
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
                (3, self.config.patch_size, self.config.patch_size),
                0.5
            )
        elif patch_type == 'random':
            adv_patch_cpu = torch.rand(
                (3, self.config.patch_size, self.config.patch_size)
            )
        else:
            raise ValueError(f"Unknown patch type: {patch_type}. Use 'gray' or 'random'.")

        return adv_patch_cpu

    def read_image(self, path: str):
        """Read an image file and convert it to a patch tensor.

        Args:
            path: Path to the image file

        Returns:
            torch.Tensor: Image tensor resized to (3, patch_size, patch_size)
        """
        patch_img = Image.open(path).convert('RGB')

        # Resize to patch size
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)

        # Convert to tensor
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)

        return adv_patch_cpu

    def save_patch(self, adv_patch_cpu, epoch: int, ep_det_loss: float = None):
        """Save the current patch to disk.

        Args:
            adv_patch_cpu: Patch tensor to save
            epoch: Current epoch number (used in filename)
            ep_det_loss: Optional detection loss value (included in filename)
        """
        im = transforms.ToPILImage('RGB')(adv_patch_cpu)

        # Ensure output directory exists
        if not os.path.exists('pics'):
            os.mkdir('pics')

        # Create filename with epoch and optional loss value
        if ep_det_loss is not None:
            filename = f'pics/{epoch}_{ep_det_loss}.png'
        else:
            filename = f'pics/{epoch}.png'

        im.save(filename, quality=100)

    def save_best_patch(self, adv_patch_cpu, epoch: int, det_loss: float):
        """Save the best performing patch so far.

        Args:
            adv_patch_cpu: Patch tensor to save
            epoch: Current epoch number
            det_loss: Detection loss value
        """
        im = transforms.ToPILImage('RGB')(adv_patch_cpu)

        # Ensure output directory exists
        if not os.path.exists('pics'):
            os.mkdir('pics')

        im.save(f'pics/best_{epoch}_{det_loss}.png', quality=100)

    def cleanup_memory(self, *tensors):
        """Clean up GPU memory by deleting tensors and clearing cache.

        Args:
            *tensors: Variable number of tensors to delete
        """
        for tensor in tensors:
            del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def train(self):
        """Train the adversarial patch.

        This method must be implemented by subclasses with dataset-specific
        training logic.
        """
        raise NotImplementedError("Subclasses must implement train() method")
