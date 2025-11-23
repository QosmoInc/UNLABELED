"""Unity dataset-specific patch trainer.

This module implements a trainer for the Unity-generated synthetic dataset with
detection loss and AdaIN style loss.
"""

import os
import time
from typing import Optional, TYPE_CHECKING

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch import autograd
from tqdm import tqdm

from .base_trainer import BasePatchTrainer
from load_data import (
    MaxProbExtractor,
    AdaINStyleLoss,
    PatchTransformer,
    PatchApplier
)
from unity_dataset import UnityDataset

if TYPE_CHECKING:
    from config_models import TrainingConfig


class UnityPatchTrainer(BasePatchTrainer):
    """Patch trainer for Unity-generated synthetic dataset.

    This trainer implements training with Unity synthetic data:
    - UnityDataset for synthetic person images
    - Dynamic dataset regeneration with current patch
    - Detection loss + AdaIN style loss
    - Custom patch size (600x600 by default)
    """

    def __init__(self, config: 'TrainingConfig', device: Optional[str] = None) -> None:
        """Initialize Unity trainer.

        Args:
            config: TrainingConfig instance with all configuration parameters
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
        """
        super().__init__(config, device)

        # Initialize Unity-specific loss functions
        # Get target class ID from config, default to person (0)
        target_class_id = self.config.target.class_id if self.config.target else 0
        self.prob_extractor: MaxProbExtractor = MaxProbExtractor(
            target_class_id, 80, self.config
        ).to(self.device)
        self.adaIN_style_loss: AdaINStyleLoss = AdaINStyleLoss().to(self.device)

    def train(self) -> None:
        """Train adversarial patch on Unity synthetic dataset.

        Implements the full training loop with:
        - Patch initialization from existing image
        - Data loading with UnityDataset
        - Dynamic dataset regeneration
        - Multi-loss optimization
        - TensorBoard logging
        - Best patch checkpointing
        """
        img_size = self.darknet_model.height
        batch_size = self.config.training.batch_size
        n_epochs = self.config.training.epochs

        # Initialize patch based on config
        if self.config.patch.initial_type == 'image' and self.config.patch.initial_image:
            orig_img = self.read_image(self.config.patch.initial_image).to(self.device)
            adv_patch_cpu = orig_img.cpu()
        elif self.config.patch.initial_type == 'gray':
            adv_patch_cpu = self.generate_patch('gray')
        elif self.config.patch.initial_type == 'random':
            adv_patch_cpu = self.generate_patch('random')
        else:
            # Default fallback to gray
            adv_patch_cpu = self.generate_patch('gray')

        # Load style image if specified
        if self.config.patch.style_image:
            orig_img = self.read_image(self.config.patch.style_image).to(self.device)
        else:
            orig_img = adv_patch_cpu.clone().to(self.device)

        adv_patch_cpu.requires_grad_(True)

        # Save initial patch
        self.save_patch(adv_patch_cpu, 0)

        # Initialize Unity dataset
        dataset = UnityDataset(
            data_dir=self.config.dataset.img_dir,
            img_size=img_size,
            num_images=614
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers
        )

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        # Initialize optimizer and scheduler
        optimizer = optim.Adam(
            [adv_patch_cpu],
            lr=self.config.training.learning_rate,
            amsgrad=True
        )
        # Use ReduceLROnPlateau scheduler (was previously from config.scheduler_factory)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

        et0 = time.time()
        best_det_loss = 1.0

        for epoch in range(1, n_epochs):
            ep_det_loss = 0
            ep_adaIN_loss = 0
            ep_loss = 0
            bt0 = time.time()

            # Regenerate dataset with current patch
            dataset.create_next_dataset(f'pics/{epoch - 1}.png')

            for i_batch, img_batch in tqdm(
                enumerate(train_loader),
                desc=f'Running epoch {epoch}',
                total=self.epoch_length
            ):
                with autograd.detect_anomaly():
                    optimizer.zero_grad()

                    img_batch = img_batch.to(self.device)
                    adv_patch = adv_patch_cpu.to(self.device)

                    # Resize image to YOLO input size
                    p_img_batch = F.interpolate(
                        img_batch,
                        (self.darknet_model.height, self.darknet_model.width)
                    )

                    # Forward pass through YOLO
                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)

                    # Calculate losses
                    det_loss = torch.mean(max_prob)
                    adaIN_loss = self.adaIN_style_loss(
                        adv_patch.unsqueeze(0),
                        orig_img.unsqueeze(0).to(self.device)
                    ) * 0.001

                    loss = det_loss + adaIN_loss

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_adaIN_loss += adaIN_loss.detach().cpu().numpy()

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Clamp patch values to valid image range
                    adv_patch_cpu.data.clamp_(0, 1)

                    bt1 = time.time()

                    # Log metrics to WandB (every 5 batches)
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        # Log scalar metrics
                        self.tracker.log_scalars({
                            'total_loss': loss,
                            'loss/det_loss': det_loss,
                            'loss/adaIN_loss': adaIN_loss,
                            'epoch': epoch,
                            'learning_rate': optimizer.param_groups[0]["lr"]
                        }, step=iteration)

                        # Log images
                        self.tracker.log_image('patch', adv_patch_cpu, step=iteration)
                        self.tracker.log_images('training_images', p_img_batch, step=iteration)

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        # Clean up memory
                        self.cleanup_memory(output, max_prob, det_loss, p_img_batch, adaIN_loss, loss)

                    bt0 = time.time()

            et1 = time.time()
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_adaIN_loss = ep_adaIN_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

            # Save patch for this epoch
            self.save_patch(adv_patch_cpu, epoch, ep_det_loss)

            # Log epoch-level metrics
            self.tracker.log_scalars({
                'epoch/det_loss': ep_det_loss,
                'epoch/adaIN_loss': ep_adaIN_loss,
                'epoch/total_loss': ep_loss,
                'epoch/time': et1 - et0
            }, step=epoch)

            # Log epoch-level images
            self.tracker.log_image('epoch/patch', adv_patch_cpu, step=epoch)
            self.tracker.log_images('epoch/training_images', p_img_batch, step=epoch)

            # Save best patch
            if det_loss.detach().cpu().numpy() < best_det_loss:
                best_det_loss = det_loss.detach().cpu().numpy()
                self.save_best_patch(adv_patch_cpu, epoch, best_det_loss)

            # Update scheduler
            scheduler.step(ep_loss)

            # Print epoch statistics
            print('  EPOCH NR: ', epoch)
            print('EPOCH LOSS: ', ep_loss)
            print('  DET LOSS: ', ep_det_loss)
            print('ADAIN LOSS: ', ep_adaIN_loss)
            print('EPOCH TIME: ', et1 - et0)

            # Final cleanup
            self.cleanup_memory(output, max_prob, det_loss, p_img_batch, adaIN_loss, loss)

            et0 = time.time()

        # Finish experiment tracking
        self.tracker.finish()
