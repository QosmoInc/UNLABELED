"""Multi-class patch trainer for simultaneous targeting of multiple classes.

This module implements a trainer that can suppress one class (e.g., person)
while enhancing another class (e.g., bear) detection simultaneously.

COCO class IDs:
- 0: person
- 21: bear
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
    TotalVariation,
    PatchTransformer,
    PatchApplier
)

if TYPE_CHECKING:
    from config_models import TrainingConfig


class MultiClassPatchTrainer(BasePatchTrainer):
    """Patch trainer for multi-class adversarial patches.

    This trainer generates patches that suppress detection of one class
    (suppress_class_id) while enhancing detection of another class
    (enhance_class_id). Useful for creating patches that make a person
    appear as a different object.
    """

    def __init__(
        self,
        config: 'TrainingConfig',
        suppress_class_id: int = 0,  # person
        enhance_class_id: int = 21,  # bear
        device: Optional[str] = None
    ) -> None:
        """Initialize MultiClass trainer.

        Args:
            config: TrainingConfig instance with all configuration parameters
            suppress_class_id: Class ID to suppress (default: 0 for person)
            enhance_class_id: Class ID to enhance (default: 21 for bear)
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
        """
        super().__init__(config, device)

        # Store class IDs
        self.suppress_class_id = suppress_class_id
        self.enhance_class_id = enhance_class_id

        # Initialize extractors for both classes
        # COCO dataset has 80 classes total
        self.prob_extractor_suppress: MaxProbExtractor = MaxProbExtractor(
            suppress_class_id, 80, self.config
        ).to(self.device)
        self.prob_extractor_enhance: MaxProbExtractor = MaxProbExtractor(
            enhance_class_id, 80, self.config
        ).to(self.device)

        # Initialize other loss functions
        self.adaIN_style_loss: AdaINStyleLoss = AdaINStyleLoss().to(self.device)
        self.content_loss: ContentLoss = ContentLoss().to(self.device)
        self.total_variation: TotalVariation = TotalVariation().to(self.device)

    def train(
        self,
        style_image_path: Optional[str] = None,
        content_image_path: Optional[str] = None
    ) -> None:
        """Train adversarial patch with multi-class objectives.

        Args:
            style_image_path: Path to style reference image (overrides config)
            content_image_path: Path to content reference image (overrides config)
        """
        img_size = self.config.patch.size
        batch_size = self.config.training.batch_size
        n_epochs = self.config.training.epochs
        max_lab = self.config.dataset.max_labels

        # Use config values if not overridden
        if style_image_path is None:
            style_image_path = self.config.patch.style_image or 'imgs/210825_ダギング_6.jpg'
        if content_image_path is None:
            content_image_path = self.config.patch.content_image or 'imgs/bear2.jpg'

        # Load style and content images
        orig_img_style = self.read_image(style_image_path).to(self.device)
        orig_img = self.read_image(content_image_path).to(self.device)

        # Initialize patch
        if self.config.patch.initial_type == 'gray':
            adv_patch_cpu = self.generate_patch("gray")
        elif self.config.patch.initial_type == 'random':
            adv_patch_cpu = self.generate_patch("random")
        elif self.config.patch.initial_type == 'image':
            if self.config.patch.initial_image:
                adv_patch_cpu = self.read_image(self.config.patch.initial_image)
            else:
                raise ValueError("initial_image must be specified when initial_type='image'")
        else:
            adv_patch_cpu = self.generate_patch("gray")

        adv_patch_cpu.requires_grad_(True)

        # Save initial patch
        self.save_patch(adv_patch_cpu, 0)

        # Initialize dataset
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
        print(f'One epoch is {len(train_loader)}')

        # Initialize optimizer and scheduler
        optimizer = optim.Adam(
            [adv_patch_cpu],
            lr=self.config.training.learning_rate,
            amsgrad=True
        )
        # Use ReduceLROnPlateau scheduler (was previously from config.scheduler_factory)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

        # Initialize patch transformer and applier
        patch_transformer = PatchTransformer().to(self.device)
        patch_applier = PatchApplier().to(self.device)

        best_suppress_loss = float('inf')

        for epoch in range(1, n_epochs + 1):
            ep_suppress_loss = 0
            ep_enhance_loss = 0
            ep_adain_loss = 0
            ep_content_loss = 0
            ep_tv_loss = 0
            ep_loss = 0

            for i_batch, (img_batch, lab_batch) in tqdm(
                enumerate(train_loader),
                desc=f'Running epoch {epoch}',
                total=self.epoch_length
            ):
                with autograd.detect_anomaly():
                    optimizer.zero_grad()

                    img_batch = img_batch.to(self.device)
                    lab_batch = lab_batch.to(self.device)
                    adv_patch = adv_patch_cpu.to(self.device)

                    # Apply transformations
                    adv_batch_t = patch_transformer(
                        adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False
                    )
                    p_img_batch = patch_applier(img_batch, adv_batch_t)

                    # Resize to YOLO input size
                    p_img_batch = F.interpolate(
                        p_img_batch,
                        (self.darknet_model.height, self.darknet_model.width)
                    )

                    # Forward pass through YOLO
                    output = self.darknet_model(p_img_batch)

                    # Calculate detection losses for both classes
                    max_prob_suppress = self.prob_extractor_suppress(output)
                    max_prob_enhance = self.prob_extractor_enhance(output)

                    # Suppress one class, enhance another
                    suppress_loss = torch.mean(max_prob_suppress)  # minimize
                    enhance_loss = -torch.mean(max_prob_enhance)  # maximize (negate to minimize)

                    # Calculate style and content losses
                    adain_loss = self.adaIN_style_loss(
                        adv_patch.unsqueeze(0),
                        orig_img_style.unsqueeze(0)
                    ) * self.config.losses.adain_weight

                    content_loss = self.content_loss(
                        adv_patch.unsqueeze(0),
                        orig_img.unsqueeze(0)
                    ) * self.config.losses.content_weight

                    # Calculate total variation loss
                    tv_loss = self.total_variation(adv_patch) * self.config.losses.tv_weight
                    tv_loss = torch.clamp(tv_loss, max=self.config.losses.tv_max)

                    # Combine all losses
                    loss = suppress_loss + enhance_loss + adain_loss + content_loss + tv_loss

                    # Track losses
                    ep_suppress_loss += suppress_loss.detach().cpu().item()
                    ep_enhance_loss += enhance_loss.detach().cpu().item()
                    ep_adain_loss += adain_loss.detach().cpu().item()
                    ep_content_loss += content_loss.detach().cpu().item()
                    ep_tv_loss += tv_loss.detach().cpu().item()
                    ep_loss += loss.detach().cpu().item()

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Clamp patch values
                    adv_patch_cpu.data.clamp_(0, 1)

                    # TensorBoard logging
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().item(), iteration)
                        self.writer.add_scalar('loss/suppress_loss', suppress_loss.detach().cpu().item(), iteration)
                        self.writer.add_scalar('loss/enhance_loss', enhance_loss.detach().cpu().item(), iteration)
                        self.writer.add_scalar('loss/adain_loss', adain_loss.detach().cpu().item(), iteration)
                        self.writer.add_scalar('loss/content_loss', content_loss.detach().cpu().item(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().item(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                        self.writer.add_image('training_images', torchvision.utils.make_grid(p_img_batch), iteration)

                    if i_batch + 1 < len(train_loader):
                        # Clean up memory
                        self.cleanup_memory(
                            output, max_prob_suppress, max_prob_enhance,
                            suppress_loss, enhance_loss, adain_loss,
                            content_loss, tv_loss, loss, p_img_batch
                        )

            # Calculate average losses
            ep_suppress_loss /= len(train_loader)
            ep_enhance_loss /= len(train_loader)
            ep_adain_loss /= len(train_loader)
            ep_content_loss /= len(train_loader)
            ep_tv_loss /= len(train_loader)
            ep_loss /= len(train_loader)

            # Save patch for this epoch
            self.save_patch(adv_patch_cpu, epoch)

            # Save best patch (based on suppress loss)
            if ep_suppress_loss < best_suppress_loss:
                best_suppress_loss = ep_suppress_loss
                self.save_best_patch(adv_patch_cpu, epoch, best_suppress_loss)

            # Update scheduler
            scheduler.step(ep_loss)

            # Print epoch statistics
            print(f'  EPOCH NR: {epoch}')
            print(f'EPOCH LOSS: {ep_loss:.4f}')
            print(f'SUPPRESS LOSS (class {self.suppress_class_id}): {ep_suppress_loss:.4f}')
            print(f'ENHANCE LOSS (class {self.enhance_class_id}): {ep_enhance_loss:.4f}')
            print(f'ADAIN LOSS: {ep_adain_loss:.4f}')
            print(f'CONTENT LOSS: {ep_content_loss:.4f}')
            print(f'TV LOSS: {ep_tv_loss:.4f}')

        # Close TensorBoard writer
        self.writer.close()
