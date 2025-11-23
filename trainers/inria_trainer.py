"""INRIA dataset-specific patch trainer.

This module implements a trainer for the INRIA Person Dataset with
multiple loss functions including detection loss, style loss, content loss,
and total variation loss.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch import autograd
from tqdm import tqdm

from .base_trainer import BasePatchTrainer
from load_data import (
    InriaDataset,
    MaxProbExtractor,
    AdaINStyleLoss,
    ContentLoss,
    TotalVariation
)


class InriaPatchTrainer(BasePatchTrainer):
    """Trainer for adversarial patches using INRIA Person Dataset.

    This trainer implements the full training pipeline with:
    - INRIA Person Dataset for realistic person images
    - YOLOv2 MS COCO weights for person detection
    - Multiple loss functions: detection, AdaIN style, content, total variation
    - TensorBoard logging and checkpoint saving
    """

    def __init__(self, mode: str, device: str = None):
        """Initialize INRIA trainer.

        Args:
            mode: Configuration name from patch_config.patch_configs
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
        """
        super().__init__(mode, device)

        # Initialize INRIA-specific loss functions
        # Person class ID is 0 in COCO dataset (80 classes total)
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).to(self.device)
        self.adaIN_style_loss = AdaINStyleLoss().to(self.device)
        self.content_loss = ContentLoss().to(self.device)
        self.total_variation = TotalVariation().to(self.device)

    def train(self):
        """Train adversarial patch on INRIA Person Dataset.

        Implements the full training loop with:
        - Patch initialization (gray or from image)
        - Data loading with INRIA dataset
        - Multi-loss optimization
        - TensorBoard logging
        - Best patch checkpointing
        """
        # Training hyperparameters
        img_size = self.config.patch_size
        batch_size = self.config.batch_size
        n_epochs = 10000
        max_lab = 14  # Maximum number of persons per image

        # Generate initial patch
        adv_patch_cpu = self.generate_patch("gray")
        # Alternative: Load patch from image
        # adv_patch_cpu = self.read_image('imgs/01.png')

        # Load style and content reference images
        orig_img = self.read_image('imgs/2025_11_22/01-1.jpg').to(self.device)
        orig_img_style = self.read_image('imgs/2025_11_22/01-1.jpg').to(self.device)

        # Enable gradient computation for patch
        adv_patch_cpu.requires_grad_(True)

        # Save initial patch
        self.save_patch(adv_patch_cpu, 0, 0)

        # Setup dataset and dataloader
        dataset = InriaDataset(
            self.config.img_dir,
            self.config.lab_dir,
            max_lab,
            img_size,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)} batches')

        # Setup optimizer and scheduler
        optimizer = optim.Adam(
            [adv_patch_cpu],
            lr=self.config.start_learning_rate,
            amsgrad=True
        )
        scheduler = self.config.scheduler_factory(optimizer)

        # Training loop
        best_det_loss = 1.0

        for epoch in range(1, n_epochs):
            # Epoch metrics
            ep_det_loss = 0
            ep_adaIN_loss = 0
            ep_c_loss = 0
            ep_loss = 0

            # Batch loop with progress bar
            for i_batch, (img_batch, lab_batch) in tqdm(
                enumerate(train_loader),
                desc=f'Running epoch {epoch}',
                total=self.epoch_length
            ):
                with autograd.detect_anomaly():
                    optimizer.zero_grad()

                    # Move data to device
                    img_batch = img_batch.to(self.device)
                    lab_batch = lab_batch.to(self.device)
                    adv_patch = adv_patch_cpu.to(self.device)

                    # Transform and apply patches
                    adv_batch_t = self.patch_transformer(
                        adv_patch,
                        lab_batch,
                        img_size,
                        do_rotate=True,
                        rand_loc=False
                    )
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                    # Resize to YOLO input size
                    p_img_batch = F.interpolate(
                        p_img_batch,
                        (self.darknet_model.height, self.darknet_model.width)
                    )

                    # Forward pass through detection model
                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)

                    # Calculate losses
                    # Detection loss: minimize person detection confidence
                    det_loss = torch.mean(max_prob)

                    # Style loss: match style of reference image (weight: 0.0 = disabled)
                    adaIN_loss = self.adaIN_style_loss(
                        adv_patch.unsqueeze(0),
                        orig_img_style.unsqueeze(0).to(self.device)
                    ) * 0.0

                    # Content loss: preserve content structure (weight: 5.0)
                    c_loss = self.content_loss(adv_patch, orig_img) * 5.0

                    # Total variation loss: encourage smoothness (weight: 0.5, max: 0.1)
                    tv_loss = self.total_variation(adv_patch) * 0.5

                    # Combined loss
                    loss = (
                        det_loss +
                        adaIN_loss +
                        c_loss +
                        torch.max(tv_loss, torch.tensor(0.1).to(self.device))
                    )

                    # Accumulate epoch metrics
                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_adaIN_loss += adaIN_loss.detach().cpu().numpy()
                    ep_c_loss += c_loss.detach().cpu().numpy()

                    # Backpropagation
                    loss.backward()
                    optimizer.step()

                    # Clamp patch values to valid image range [0, 1]
                    adv_patch_cpu.data.clamp_(0, 1)

                    # TensorBoard logging (every 5 batches)
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/adaIN_loss', adaIN_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/content_loss', c_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                        self.writer.add_image('training_images', torchvision.utils.make_grid(p_img_batch), iteration)

                    # Clean up memory
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        self.cleanup_memory(output, max_prob, det_loss, p_img_batch, adaIN_loss, c_loss, loss)

            # Calculate epoch averages
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_adaIN_loss = ep_adaIN_loss / len(train_loader)
            ep_c_loss = ep_c_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

            # Save current patch
            self.save_patch(adv_patch_cpu, epoch, ep_det_loss)

            # Add epoch-level visualizations to TensorBoard
            self.writer.add_image('patch', adv_patch_cpu, epoch)
            self.writer.add_image('training_images', torchvision.utils.make_grid(p_img_batch), epoch)

            # Save best patch
            if det_loss.detach().cpu().numpy() < best_det_loss:
                best_det_loss = det_loss.detach().cpu().numpy()
                self.save_best_patch(adv_patch_cpu, epoch, det_loss.detach().cpu().numpy())

            # Step learning rate scheduler
            scheduler.step(ep_loss)

            # Print epoch summary
            print('  EPOCH NR: ', epoch)
            print('EPOCH LOSS: ', ep_loss)
            print('  DET LOSS: ', ep_det_loss)
            print('ADAIN LOSS: ', ep_adaIN_loss)
            print('    C LOSS: ', ep_c_loss)

            # Final cleanup
            self.cleanup_memory(output, max_prob, det_loss, p_img_batch, adaIN_loss, c_loss, loss)

        # Close TensorBoard writer
        self.writer.close()
