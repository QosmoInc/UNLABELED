"""Unity dataset-specific patch trainer.

This module implements a trainer for the Unity-generated synthetic dataset with
detection loss and AdaIN style loss.
"""

import os
import time
from typing import Optional

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


class UnityPatchTrainer(BasePatchTrainer):
    """Patch trainer for Unity-generated synthetic dataset.

    This trainer implements training with Unity synthetic data:
    - UnityDataset for synthetic person images
    - Dynamic dataset regeneration with current patch
    - Detection loss + AdaIN style loss
    - Custom patch size (600x600 by default)
    """

    def __init__(self, mode: str, device: Optional[str] = None) -> None:
        """Initialize Unity trainer.

        Args:
            mode: Configuration name from patch_config.patch_configs
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
        """
        super().__init__(mode, device)

        # Override patch size for Unity dataset
        self.config.patch_size = 600

        # Initialize Unity-specific loss functions
        # Person class ID is 0 in COCO dataset (80 classes total)
        self.prob_extractor: MaxProbExtractor = MaxProbExtractor(0, 80, self.config).to(self.device)
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
        batch_size = self.config.batch_size
        n_epochs = 10000

        # Generate starting point from existing image
        orig_img = self.read_image('imgs/AF_patch_mayuu_01.jpg').to(self.device)
        adv_patch_cpu = orig_img.cpu()
        adv_patch_cpu.requires_grad_(True)

        # Save initial patch
        self.save_patch(adv_patch_cpu, 0)

        # Initialize Unity dataset
        dataset = UnityDataset(data_dir='train_data_0', img_size=img_size, num_images=614)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        # Initialize optimizer and scheduler
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

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

                    # TensorBoard logging
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/adaIN_loss', adaIN_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                        self.writer.add_image('training_images', torchvision.utils.make_grid(p_img_batch), iteration)

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
            self.save_patch(adv_patch_cpu, epoch)

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

        # Close TensorBoard writer
        self.writer.close()
