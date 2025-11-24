"""Configuration models using Pydantic v2 for type-safe YAML configuration.

This module provides Pydantic models for validating and managing training configurations.
All configurations are YAML-based with full type safety and validation.

Features:
- Type-safe configuration with automatic validation
- File path existence checks
- Range validation for numerical parameters
- Enum-based choices for string parameters
- Default values for optional parameters

Example:
    # Load from YAML
    config = TrainingConfig.from_yaml('configs/person_inria.yaml')

    # Validate all paths and parameters
    config.validate()

    # Access typed fields
    print(config.training.batch_size)  # IDE autocomplete works!

    # Save modified config
    config.to_yaml('configs/modified.yaml')
"""

from __future__ import annotations

import warnings
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# Enums for type-safe string choices
class TrainerType(str, Enum):
    """Available trainer types."""
    INRIA = "InriaPatchTrainer"
    CAT = "CatPatchTrainer"
    DOG = "DogPatchTrainer"
    UNITY = "UnityPatchTrainer"
    MULTICLASS = "MultiClassPatchTrainer"


class DatasetType(str, Enum):
    """Available dataset types."""
    INRIA = "inria"
    UNITY = "unity"


class PatchInitType(str, Enum):
    """Patch initialization methods."""
    GRAY = "gray"
    RANDOM = "random"
    IMAGE = "image"  # Load from file


class Objective(str, Enum):
    """Detection objective."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# Configuration sub-models
class TrainerConfig(BaseModel):
    """Trainer configuration."""

    model_config = ConfigDict(use_enum_values=True, extra='allow')

    type: TrainerType = Field(
        description="Type of trainer to use"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device to use (e.g., 'cuda:0', 'cpu'). None for auto-detect."
    )


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    model_config = ConfigDict(use_enum_values=True, extra='allow')

    type: DatasetType = Field(
        default=DatasetType.INRIA,
        description="Type of dataset"
    )
    img_dir: str = Field(
        description="Directory containing images"
    )
    lab_dir: str = Field(
        description="Directory containing YOLO label files"
    )
    max_labels: Annotated[int, Field(gt=0, le=50)] = Field(
        default=14,
        description="Maximum number of labels per image"
    )

    @field_validator('img_dir', 'lab_dir')
    @classmethod
    def validate_directory(cls, v: str) -> str:
        """Validate directory path (warning only during construction)."""
        # We only warn here, actual existence check happens in validate()
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    model_config = ConfigDict(extra='allow')

    cfgfile: str = Field(
        description="Path to YOLO configuration file"
    )
    weightfile: str = Field(
        description="Path to YOLO weights file"
    )


class PatchConfig(BaseModel):
    """Patch configuration."""

    model_config = ConfigDict(use_enum_values=True, extra='allow')

    size: Annotated[int, Field(gt=0, le=2000)] = Field(
        default=300,
        description="Patch size in pixels (square)"
    )
    name: str = Field(
        default="patch",
        description="Name for this patch (used in output filenames)"
    )
    initial_type: PatchInitType = Field(
        default=PatchInitType.GRAY,
        description="How to initialize the patch"
    )
    initial_image: Optional[str] = Field(
        default=None,
        description="Path to image for initialization (if initial_type='image')"
    )
    style_image: Optional[str] = Field(
        default=None,
        description="Path to style image for style transfer loss"
    )
    content_image: Optional[str] = Field(
        default=None,
        description="Path to content image for content loss"
    )


class TrainingParams(BaseModel):
    """Training hyperparameters."""

    model_config = ConfigDict(extra='allow')

    batch_size: Annotated[int, Field(gt=0, le=128)] = Field(
        default=8,
        description="Batch size for training"
    )
    epochs: Annotated[int, Field(gt=0, le=10000)] = Field(
        default=300,
        description="Number of training epochs"
    )
    learning_rate: Annotated[float, Field(gt=0.0, le=1.0)] = Field(
        default=0.03,
        description="Initial learning rate"
    )
    num_workers: Annotated[int, Field(ge=0, le=32)] = Field(
        default=4,
        description="Number of data loader workers"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. None for non-deterministic training."
    )


class LossWeights(BaseModel):
    """Loss function weights."""

    model_config = ConfigDict(extra='allow')

    detection_weight: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Weight for detection loss"
    )
    adain_weight: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Weight for AdaIN style loss"
    )
    content_weight: Annotated[float, Field(ge=0.0)] = Field(
        default=5.0,
        description="Weight for content loss"
    )
    tv_weight: Annotated[float, Field(ge=0.0)] = Field(
        default=0.5,
        description="Weight for total variation loss"
    )
    tv_max: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.165,
        description="Maximum total variation value"
    )


class TargetConfig(BaseModel):
    """Target class configuration."""

    model_config = ConfigDict(use_enum_values=True, extra='allow')

    class_id: Annotated[int, Field(ge=0, lt=80)] = Field(
        default=0,
        description="Target class ID (COCO dataset)"
    )
    objective: Objective = Field(
        default=Objective.MINIMIZE,
        description="Whether to minimize or maximize detection"
    )


class MultiClassTargets(BaseModel):
    """Multi-class target configuration."""

    model_config = ConfigDict(extra='allow')

    suppress: TargetConfig = Field(
        description="Class to suppress"
    )
    enhance: TargetConfig = Field(
        description="Class to enhance"
    )


class WandBConfig(BaseModel):
    """Weights & Biases configuration."""

    model_config = ConfigDict(extra='allow')

    enabled: bool = Field(
        default=True,
        description="Enable WandB tracking"
    )
    project: str = Field(
        default="adversarial-patch",
        description="WandB project name"
    )
    entity: Optional[str] = Field(
        default=None,
        description="WandB entity/username"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for this run"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Notes/description for this run"
    )


class LoggingConfig(BaseModel):
    """Logging configuration for unified logging system."""

    model_config = ConfigDict(extra='allow')

    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    console: bool = Field(
        default=True,
        description="Enable console logging"
    )
    file: bool = Field(
        default=False,
        description="Enable file logging"
    )
    log_dir: str = Field(
        default="logs",
        description="Directory for log files"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid log level: {v}. Must be one of {valid_levels}"
            )
        return v_upper


class TrainingConfig(BaseModel):
    """Complete training configuration.

    This is the root configuration model that contains all sub-configurations.
    """

    model_config = ConfigDict(extra='allow', validate_assignment=True)

    trainer: TrainerConfig = Field(
        description="Trainer configuration"
    )
    dataset: DatasetConfig = Field(
        description="Dataset configuration"
    )
    model: ModelConfig = Field(
        description="Model configuration"
    )
    patch: PatchConfig = Field(
        default_factory=PatchConfig,
        description="Patch configuration"
    )
    training: TrainingParams = Field(
        default_factory=TrainingParams,
        description="Training hyperparameters"
    )
    losses: LossWeights = Field(
        default_factory=LossWeights,
        description="Loss function weights"
    )
    target: Optional[TargetConfig] = Field(
        default=None,
        description="Single target configuration (for non-multiclass trainers)"
    )
    targets: Optional[MultiClassTargets] = Field(
        default=None,
        description="Multi-class targets (for MultiClassPatchTrainer)"
    )
    wandb: WandBConfig = Field(
        default_factory=WandBConfig,
        description="Weights & Biases configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )

    @model_validator(mode='after')
    def validate_trainer_requirements(self) -> 'TrainingConfig':
        """Validate trainer-specific requirements."""
        # MultiClassPatchTrainer requires 'targets'
        if self.trainer.type == TrainerType.MULTICLASS:
            if self.targets is None:
                raise ValueError(
                    "MultiClassPatchTrainer requires 'targets' configuration"
                )
        else:
            # Other trainers should have 'target'
            if self.target is None:
                # Provide default target
                self.target = TargetConfig()

        # Validate patch initialization
        if self.patch.initial_type == PatchInitType.IMAGE:
            if self.patch.initial_image is None:
                raise ValueError(
                    "initial_image must be specified when initial_type='image'"
                )

        return self

    def validate(self) -> None:
        """Perform runtime validation checks.

        This method performs checks that should happen before training,
        such as verifying that all required files and directories exist.

        Raises:
            FileNotFoundError: If required files/directories don't exist
            ValueError: If configuration is invalid
        """
        # Check critical files exist
        critical_files = [
            ('Model config', self.model.cfgfile),
            ('Model weights', self.model.weightfile),
        ]

        missing_files = []
        for name, path in critical_files:
            if not Path(path).exists():
                missing_files.append(f"{name}: {path}")

        if missing_files:
            raise FileNotFoundError(
                "Missing required files:\n  " + "\n  ".join(missing_files)
            )

        # Check dataset directories
        if not Path(self.dataset.img_dir).exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.dataset.img_dir}"
            )

        if not Path(self.dataset.lab_dir).exists():
            raise FileNotFoundError(
                f"Label directory does not exist: {self.dataset.lab_dir}"
            )

        # Check optional image files if specified
        for field_name, field_path in [
            ('initial_image', self.patch.initial_image),
            ('style_image', self.patch.style_image),
            ('content_image', self.patch.content_image),
        ]:
            if field_path is not None and not Path(field_path).exists():
                warnings.warn(
                    f"Patch {field_name} does not exist: {field_path}",
                    UserWarning
                )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'TrainingConfig':
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid or validation fails
        """
        yaml_file = Path(yaml_path)

        if not yaml_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_file, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse YAML: {e}")

        if data is None:
            raise ValueError(f"Empty YAML file: {yaml_path}")

        # Create and validate config
        try:
            config = cls(**data)
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")

        return config

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_file = Path(yaml_path)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and dump
        data = self.model_dump(mode='python', exclude_none=True)

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return self.model_dump(mode='python', exclude_none=True)
