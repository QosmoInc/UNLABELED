"""INRIA dataset-specific patch trainer.

This module implements a trainer for the INRIA Person Dataset with
multiple loss functions including detection loss, style loss, content loss,
and total variation loss.
"""

from typing import Optional, TYPE_CHECKING

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

if TYPE_CHECKING:
    from config_models import TrainingConfig


class InriaPatchTrainer(BasePatchTrainer):
    """Trainer for adversarial patches using INRIA Person Dataset.

    This trainer implements the full training pipeline with:
    - INRIA Person Dataset for realistic person images
    - YOLOv2 MS COCO weights for person detection
    - Multiple loss functions: detection, AdaIN style, content, total variation
    - TensorBoard logging and checkpoint saving
    """

    def __init__(self, config: 'TrainingConfig', device: Optional[str] = None) -> None:
        """Initialize INRIA trainer.

        Args:
            config: TrainingConfig instance with all configuration parameters
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
        """
        super().__init__(config, device)

        # Initialize INRIA-specific loss functions
        # Get target class ID from config, default to 0 (person class in COCO)
        target_class_id = self.config.target.class_id if self.config.target else 0
        num_classes = 80  # COCO dataset has 80 classes

        self.prob_extractor: MaxProbExtractor = MaxProbExtractor(target_class_id, num_classes, self.config).to(self.device)
        self.adaIN_style_loss: AdaINStyleLoss = AdaINStyleLoss().to(self.device)
        self.content_loss: ContentLoss = ContentLoss().to(self.device)
        self.total_variation: TotalVariation = TotalVariation().to(self.device)

    def train(self) -> None:
        """Train adversarial patch on INRIA Person Dataset.

        Implements the full training loop with:
        - Patch initialization (gray or from image)
        - Data loading with INRIA dataset
        - Multi-loss optimization
        - TensorBoard logging
        - Best patch checkpointing
        """
        # Training hyperparameters from config
        img_size = self.config.patch.size
        batch_size = self.config.training.batch_size
        n_epochs = self.config.training.epochs
        max_lab = self.config.dataset.max_labels

        # Generate initial patch based on config
        if self.config.patch.initial_type == "image" and self.config.patch.initial_image:
            adv_patch_cpu = self.read_image(self.config.patch.initial_image)
        else:
            # Default to gray or use the configured initial_type
            adv_patch_cpu = self.generate_patch(self.config.patch.initial_type)

        # Load style and content reference images from config
        # Default to None if not specified
        orig_img = None
        orig_img_style = None

        if self.config.patch.content_image:
            orig_img = self.read_image(self.config.patch.content_image).to(self.device)

        if self.config.patch.style_image:
            orig_img_style = self.read_image(self.config.patch.style_image).to(self.device)

        # Enable gradient computation for patch
        adv_patch_cpu.requires_grad_(True)

        # Save initial patch
        self.save_patch(adv_patch_cpu, 0, 0)

        # Setup dataset and dataloader
        dataset = InriaDataset(
            self.config.dataset.img_dir,
            self.config.dataset.lab_dir,
            max_lab,
            img_size,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers
        )

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)} batches')

        # Setup optimizer
        optimizer = optim.Adam(
            [adv_patch_cpu],
            lr=self.config.training.learning_rate,
            amsgrad=True
        )
        # Note: scheduler_factory is not in the new TrainingConfig
        # Using ReduceLROnPlateau as a sensible default
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

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
                    det_loss = torch.mean(max_prob) * self.config.losses.detection_weight

                    # Style loss: match style of reference image
                    if orig_img_style is not None and self.config.losses.adain_weight > 0:
                        adaIN_loss = self.adaIN_style_loss(
                            adv_patch.unsqueeze(0),
                            orig_img_style.unsqueeze(0).to(self.device)
                        ) * self.config.losses.adain_weight
                    else:
                        adaIN_loss = torch.tensor(0.0).to(self.device)

                    # Content loss: preserve content structure
                    if orig_img is not None and self.config.losses.content_weight > 0:
                        c_loss = self.content_loss(adv_patch, orig_img) * self.config.losses.content_weight
                    else:
                        c_loss = torch.tensor(0.0).to(self.device)

                    # Total variation loss: encourage smoothness
                    tv_loss = self.total_variation(adv_patch) * self.config.losses.tv_weight

                    # Combined loss
                    loss = (
                        det_loss +
                        adaIN_loss +
                        c_loss +
                        torch.max(tv_loss, torch.tensor(self.config.losses.tv_max).to(self.device))
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
