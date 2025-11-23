"""Adversarial patch evaluation tools.

This module provides utilities to evaluate the effectiveness of adversarial
patches by measuring detection probabilities and recognition rates.
"""

from typing import Optional, Dict, List, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from darknet import Darknet
from load_data import (
    InriaDataset,
    PatchTransformer,
    PatchApplier,
    MaxProbExtractor
)
import patch_config


class PatchEvaluator:
    """Evaluator for adversarial patch effectiveness.

    This class evaluates how well adversarial patches reduce detection
    confidence for a target object class.

    Attributes:
        config: Patch configuration
        darknet_model: Loaded YOLO model
        patch_applier: Patch application module
        patch_transformer: Patch transformation module
        prob_extractor: Probability extraction module
        device: Compute device (cuda/cpu)
    """

    def __init__(
        self,
        mode: str,
        target_class: int = 0,
        num_classes: int = 80,
        device: Optional[str] = None
    ) -> None:
        """Initialize the patch evaluator.

        Args:
            mode: Configuration name from patch_config.patch_configs
            target_class: Target class ID to suppress (default: 0 for person)
            num_classes: Total number of classes in dataset (default: 80 for COCO)
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
        """
        self.config = patch_config.patch_configs[mode]()

        # Setup device
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f'Using device: {self.device}')
        print(self.config)
        print('=' * 60)

        # Load YOLO model
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().to(self.device)

        # Initialize patch processing modules
        self.patch_applier = PatchApplier().to(self.device)
        self.patch_transformer = PatchTransformer().to(self.device)
        self.prob_extractor = MaxProbExtractor(
            target_class, num_classes, self.config
        ).to(self.device)

    def load_patch(self, patch_path: str, patch_size: int = 300) -> torch.Tensor:
        """Load an adversarial patch from file.

        Args:
            patch_path: Path to patch image file
            patch_size: Size to resize patch to (default: 300x300)

        Returns:
            Patch tensor in [C, H, W] format
        """
        patch_img = Image.open(patch_path).convert('RGB')
        tf_resize = transforms.Resize((patch_size, patch_size))
        patch_img = tf_resize(patch_img)
        tf_tensor = transforms.ToTensor()
        patch_tensor = tf_tensor(patch_img)

        return patch_tensor

    def evaluate_patch_on_dataset(
        self,
        patch_path: str,
        patch_size: int = 300,
        batch_size: int = 8,
        max_lab: int = 14,
        num_batches: Optional[int] = None,
        do_rotate: bool = True,
        rand_loc: bool = False
    ) -> Dict[str, float]:
        """Evaluate patch effectiveness on INRIA dataset.

        Args:
            patch_path: Path to adversarial patch image
            patch_size: Size of patch (default: 300)
            batch_size: Batch size for evaluation (default: 8)
            max_lab: Maximum number of labels per image (default: 14)
            num_batches: Number of batches to evaluate (None = all)
            do_rotate: Whether to apply rotation augmentation
            rand_loc: Whether to apply random location

        Returns:
            Dictionary with evaluation metrics:
                - mean_prob: Mean detection probability with patch
                - std_prob: Standard deviation of detection probability
                - min_prob: Minimum detection probability
                - max_prob: Maximum detection probability
                - num_samples: Number of samples evaluated
        """
        # Load patch
        patch = self.load_patch(patch_path, patch_size).to(self.device)

        # Create dataset
        dataset = InriaDataset(
            self.config.img_dir,
            self.config.lab_dir,
            max_lab,
            patch_size,
            shuffle=False
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Evaluate
        all_probs = []
        total_batches = num_batches if num_batches is not None else len(loader)

        with torch.no_grad():
            for i_batch, (img_batch, lab_batch) in enumerate(tqdm(
                loader,
                desc='Evaluating patch',
                total=total_batches
            )):
                if num_batches is not None and i_batch >= num_batches:
                    break

                img_batch = img_batch.to(self.device)
                lab_batch = lab_batch.to(self.device)

                # Apply patch transformations
                patch_batch = self.patch_transformer(
                    patch, lab_batch, patch_size,
                    do_rotate=do_rotate, rand_loc=rand_loc
                )
                patched_img_batch = self.patch_applier(img_batch, patch_batch)

                # Resize to model input size
                patched_img_batch = F.interpolate(
                    patched_img_batch,
                    (self.darknet_model.height, self.darknet_model.width)
                )

                # Run detection
                output = self.darknet_model(patched_img_batch)
                max_prob = self.prob_extractor(output)

                # Store probabilities
                all_probs.extend(max_prob.cpu().numpy().tolist())

        # Compute statistics
        all_probs = np.array(all_probs)
        results = {
            'mean_prob': float(np.mean(all_probs)),
            'std_prob': float(np.std(all_probs)),
            'min_prob': float(np.min(all_probs)),
            'max_prob': float(np.max(all_probs)),
            'num_samples': len(all_probs)
        }

        return results

    def compare_with_without_patch(
        self,
        patch_path: str,
        patch_size: int = 300,
        batch_size: int = 8,
        max_lab: int = 14,
        num_batches: int = 10,
        do_rotate: bool = True,
        rand_loc: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compare detection with and without adversarial patch.

        Args:
            patch_path: Path to adversarial patch image
            patch_size: Size of patch (default: 300)
            batch_size: Batch size (default: 8)
            max_lab: Maximum labels per image (default: 14)
            num_batches: Number of batches to evaluate
            do_rotate: Whether to apply rotation
            rand_loc: Whether to use random locations

        Returns:
            Tuple of (without_patch_stats, with_patch_stats)
        """
        # Load patch
        patch = self.load_patch(patch_path, patch_size).to(self.device)

        # Create dataset
        dataset = InriaDataset(
            self.config.img_dir,
            self.config.lab_dir,
            max_lab,
            patch_size,
            shuffle=False
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        probs_without_patch = []
        probs_with_patch = []

        with torch.no_grad():
            for i_batch, (img_batch, lab_batch) in enumerate(tqdm(
                loader,
                desc='Comparing with/without patch',
                total=num_batches
            )):
                if i_batch >= num_batches:
                    break

                img_batch = img_batch.to(self.device)
                lab_batch = lab_batch.to(self.device)

                # Resize to model input size
                clean_img_batch = F.interpolate(
                    img_batch,
                    (self.darknet_model.height, self.darknet_model.width)
                )

                # Evaluate WITHOUT patch
                output_clean = self.darknet_model(clean_img_batch)
                prob_clean = self.prob_extractor(output_clean)
                probs_without_patch.extend(prob_clean.cpu().numpy().tolist())

                # Apply patch
                patch_batch = self.patch_transformer(
                    patch, lab_batch, patch_size,
                    do_rotate=do_rotate, rand_loc=rand_loc
                )
                patched_img_batch = self.patch_applier(img_batch, patch_batch)
                patched_img_batch = F.interpolate(
                    patched_img_batch,
                    (self.darknet_model.height, self.darknet_model.width)
                )

                # Evaluate WITH patch
                output_patched = self.darknet_model(patched_img_batch)
                prob_patched = self.prob_extractor(output_patched)
                probs_with_patch.extend(prob_patched.cpu().numpy().tolist())

        # Compute statistics
        without_stats = {
            'mean_prob': float(np.mean(probs_without_patch)),
            'std_prob': float(np.std(probs_without_patch)),
            'num_samples': len(probs_without_patch)
        }

        with_stats = {
            'mean_prob': float(np.mean(probs_with_patch)),
            'std_prob': float(np.std(probs_with_patch)),
            'num_samples': len(probs_with_patch)
        }

        # Compute reduction
        reduction = (without_stats['mean_prob'] - with_stats['mean_prob']) / without_stats['mean_prob'] * 100

        print(f'\n{"=" * 60}')
        print(f'Detection Probability Comparison:')
        print(f'  Without patch: {without_stats["mean_prob"]:.4f} ± {without_stats["std_prob"]:.4f}')
        print(f'  With patch:    {with_stats["mean_prob"]:.4f} ± {with_stats["std_prob"]:.4f}')
        print(f'  Reduction:     {reduction:.2f}%')
        print(f'{"=" * 60}\n')

        return without_stats, with_stats
