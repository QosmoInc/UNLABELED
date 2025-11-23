"""Experiment tracking module with WandB.

This module provides experiment tracking using Weights & Biases (WandB).

Features:
- Automatic WandB initialization with configuration tracking
- Metric logging (scalars, images, artifacts)
- Artifact versioning for patches and checkpoints
- Graceful fallback when WandB is disabled

Example:
    # Initialize tracker
    tracker = ExperimentTracker(
        config=training_config,
        enable_wandb=True
    )

    # Log metrics
    tracker.log_scalar('loss/det_loss', 0.5, step=100)
    tracker.log_image('patch', patch_tensor, step=100)

    # Save artifacts
    tracker.save_patch_artifact(patch_tensor, epoch=10, loss=0.3)

    # Close tracker
    tracker.finish()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

import torch
import torchvision
from PIL import Image

if TYPE_CHECKING:
    from config_models import TrainingConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


class ExperimentTracker:
    """Experiment tracking with WandB.

    This class provides an interface for logging experiments to WandB.
    It handles initialization, metric logging, artifact tracking, and
    graceful degradation when WandB is unavailable.
    """

    def __init__(
        self,
        config: 'TrainingConfig',
        experiment_name: Optional[str] = None,
        enable_wandb: bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        wandb_notes: Optional[str] = None,
    ) -> None:
        """Initialize experiment tracker.

        Args:
            config: TrainingConfig instance with all configuration
            experiment_name: Optional name for this experiment
            enable_wandb: Whether to enable WandB tracking
            wandb_project: WandB project name (from config if not specified)
            wandb_entity: WandB entity/username
            wandb_tags: Tags for this run
            wandb_notes: Notes/description for this run
        """
        self.config = config
        self.experiment_name = experiment_name or config.patch.name
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE

        # WandB initialization
        self.wandb_run: Optional[Any] = None
        if self.enable_wandb:
            if not WANDB_AVAILABLE:
                print('⚠ WandB not available. Install with: uv add wandb')
                print('  Continuing without experiment tracking...')
                self.enable_wandb = False
            else:
                self._init_wandb(
                    project=wandb_project,
                    entity=wandb_entity,
                    tags=wandb_tags,
                    notes=wandb_notes
                )

    def _init_wandb(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Initialize WandB tracking.

        Args:
            project: WandB project name
            entity: WandB entity/username
            tags: Tags for this run
            notes: Notes/description for this run
        """
        # Generate timestamped run name
        time_str = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{time_str}_{self.experiment_name}"

        # Prepare configuration dict for WandB
        config_dict = self.config.to_dict()

        # Add default tags
        if tags is None:
            tags = []
        tags.extend([
            self.config.trainer.type,
            self.config.dataset.type,
            f"patch_{self.config.patch.size}px"
        ])

        try:
            # Initialize WandB run
            self.wandb_run = wandb.init(
                project=project or "adversarial-patch",
                entity=entity,
                name=run_name,
                config=config_dict,
                tags=tags,
                notes=notes,
                resume='allow'
            )

            print(f'✓ WandB initialized: {self.wandb_run.url}')

        except Exception as e:
            print(f'⚠ Failed to initialize WandB: {e}')
            print('  Continuing without experiment tracking...')
            self.enable_wandb = False
            self.wandb_run = None

    def log_scalar(
        self,
        name: str,
        value: Union[float, int, torch.Tensor],
        step: Optional[int] = None
    ) -> None:
        """Log a scalar metric.

        Args:
            name: Metric name (e.g., 'loss/det_loss')
            value: Scalar value
            step: Step/iteration number
        """
        if not self.enable_wandb or self.wandb_run is None:
            return

        # Convert tensor to Python scalar
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()

        # Log to WandB
        log_dict = {name: value}
        if step is not None:
            log_dict['_step'] = step
        wandb.log(log_dict, step=step)

    def log_scalars(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: Optional[int] = None
    ) -> None:
        """Log multiple scalar metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            step: Step/iteration number
        """
        if not self.enable_wandb or self.wandb_run is None:
            return

        # Convert tensors to Python scalars
        clean_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                clean_metrics[name] = value.detach().cpu().item()
            else:
                clean_metrics[name] = value

        # Log to WandB
        wandb.log(clean_metrics, step=step)

    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, Image.Image],
        step: Optional[int] = None,
        caption: Optional[str] = None
    ) -> None:
        """Log an image.

        Args:
            name: Image name (e.g., 'patch', 'training_images')
            image: Image tensor (C, H, W) or PIL Image
            step: Step/iteration number
            caption: Optional caption for the image
        """
        if not self.enable_wandb or self.wandb_run is None:
            return

        # Log to WandB
        if isinstance(image, torch.Tensor):
            # Convert to numpy for WandB
            img_np = image.detach().cpu().permute(1, 2, 0).numpy()
            wandb.log({
                name: wandb.Image(img_np, caption=caption)
            }, step=step)
        else:
            wandb.log({
                name: wandb.Image(image, caption=caption)
            }, step=step)

    def log_images(
        self,
        name: str,
        images: torch.Tensor,
        step: Optional[int] = None,
        caption: Optional[str] = None,
        nrow: int = 8
    ) -> None:
        """Log a batch of images as a grid.

        Args:
            name: Image name
            images: Batch of images (B, C, H, W)
            step: Step/iteration number
            caption: Optional caption
            nrow: Number of images per row in grid
        """
        if not self.enable_wandb or self.wandb_run is None:
            return

        # Create image grid
        grid = torchvision.utils.make_grid(images, nrow=nrow)

        # Log to WandB
        img_np = grid.detach().cpu().permute(1, 2, 0).numpy()
        wandb.log({
            name: wandb.Image(img_np, caption=caption)
        }, step=step)

    def save_patch_artifact(
        self,
        patch: torch.Tensor,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> Path:
        """Save patch as artifact and track in WandB.

        Args:
            patch: Patch tensor (3, H, W)
            epoch: Current epoch
            loss: Current loss value
            is_best: Whether this is the best patch so far

        Returns:
            Path to saved patch file
        """
        # Create output directory
        output_dir = Path('pics')
        output_dir.mkdir(exist_ok=True)

        # Generate filename
        if is_best:
            filename = f'best_{epoch}_{loss:.6f}.png'
        else:
            filename = f'{epoch}_{loss:.6f}.png'

        filepath = output_dir / filename

        # Save patch to file
        from torchvision.transforms import ToPILImage
        im = ToPILImage('RGB')(patch.detach().cpu())
        im.save(filepath, quality=100)

        # Log to WandB as artifact
        if self.enable_wandb and self.wandb_run is not None:
            artifact = wandb.Artifact(
                name=f"patch_{self.experiment_name}",
                type="patch",
                description=f"Adversarial patch at epoch {epoch}",
                metadata={
                    'epoch': epoch,
                    'loss': loss,
                    'is_best': is_best,
                    'patch_size': self.config.patch.size,
                    'trainer_type': self.config.trainer.type
                }
            )
            artifact.add_file(str(filepath))
            self.wandb_run.log_artifact(artifact)

            # Also mark best patches specially
            if is_best:
                self.wandb_run.summary['best_epoch'] = epoch
                self.wandb_run.summary['best_loss'] = loss

        return filepath

    def save_checkpoint(
        self,
        checkpoint_dict: Dict[str, Any],
        epoch: int,
        is_best: bool = False
    ) -> Path:
        """Save checkpoint and track in WandB.

        Args:
            checkpoint_dict: Dictionary containing model state, optimizer state, etc.
            epoch: Current epoch
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint file
        """
        # Create output directory
        output_dir = Path('checkpoints')
        output_dir.mkdir(exist_ok=True)

        # Generate filename
        if is_best:
            filename = f'best_checkpoint_epoch_{epoch}.pth'
        else:
            filename = f'checkpoint_epoch_{epoch}.pth'

        filepath = output_dir / filename

        # Save checkpoint
        torch.save(checkpoint_dict, filepath)

        # Log to WandB as artifact
        if self.enable_wandb and self.wandb_run is not None:
            artifact = wandb.Artifact(
                name=f"checkpoint_{self.experiment_name}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}",
                metadata={
                    'epoch': epoch,
                    'is_best': is_best
                }
            )
            artifact.add_file(str(filepath))
            self.wandb_run.log_artifact(artifact)

        return filepath

    def finish(self) -> None:
        """Close WandB run."""
        if self.enable_wandb and self.wandb_run is not None:
            wandb.finish()
            print('✓ WandB run finished')

    def __enter__(self) -> 'ExperimentTracker':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.finish()
