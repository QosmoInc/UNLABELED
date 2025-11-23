#!/usr/bin/env python3
"""Unified training script for adversarial patch generation with YAML configuration.

This script provides a unified interface for training adversarial patches across
different datasets and target classes using type-safe YAML configuration files.

Usage:
    # Train with a configuration file
    python train_patch_unified.py --config configs/person_inria.yaml

    # Override configuration with command-line arguments
    python train_patch_unified.py --config configs/person_inria.yaml --batch-size 16

    # List available trainer types
    python train_patch_unified.py --list-trainers

    # Validate configuration without training
    python train_patch_unified.py --config configs/person_inria.yaml --validate-only

Examples:
    # INRIA person detection
    python train_patch_unified.py --config configs/person_inria.yaml

    # Cat detection
    python train_patch_unified.py --config configs/cat_inria.yaml

    # Multi-class (person suppression + bear enhancement)
    python train_patch_unified.py --config configs/person_bear_multiclass.yaml

    # Unity synthetic data
    python train_patch_unified.py --config configs/unity_person.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

import trainers
from config_models import TrainingConfig


# Trainer type mapping
TRAINER_REGISTRY = {
    'InriaPatchTrainer': trainers.InriaPatchTrainer,
    'CatPatchTrainer': trainers.CatPatchTrainer,
    'DogPatchTrainer': trainers.DogPatchTrainer,
    'MultiClassPatchTrainer': trainers.MultiClassPatchTrainer,
}

# Add UnityPatchTrainer if available (requires python-osc)
if hasattr(trainers, 'UnityPatchTrainer') and trainers.UnityPatchTrainer is not None:
    TRAINER_REGISTRY['UnityPatchTrainer'] = trainers.UnityPatchTrainer


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> None:
    """Apply command-line argument overrides to configuration.

    Args:
        config: Configuration instance to modify
        args: Parsed command-line arguments
    """
    # Override training parameters
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    if args.epochs is not None:
        config.training.epochs = args.epochs

    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate

    # Override patch parameters
    if args.patch_size is not None:
        config.patch.size = args.patch_size

    # Override device
    if args.device is not None:
        config.trainer.device = args.device


def create_trainer(config: TrainingConfig) -> trainers.BasePatchTrainer:
    """Create trainer instance from configuration.

    Args:
        config: Training configuration

    Returns:
        Trainer instance

    Raises:
        ValueError: If trainer type is unknown
    """
    trainer_type = config.trainer.type
    device = config.trainer.device

    # Get trainer class
    if trainer_type not in TRAINER_REGISTRY:
        available = ', '.join(TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Available types: {available}"
        )

    trainer_class = TRAINER_REGISTRY[trainer_type]

    # Create trainer with config
    # MultiClassPatchTrainer has additional parameters
    if trainer_type == 'MultiClassPatchTrainer':
        if config.targets is None:
            raise ValueError("MultiClassPatchTrainer requires 'targets' configuration")

        trainer = trainer_class(
            config=config,
            suppress_class_id=config.targets.suppress.class_id,
            enhance_class_id=config.targets.enhance.class_id,
            device=device
        )
    else:
        trainer = trainer_class(config=config, device=device)

    return trainer


def list_available_trainers() -> None:
    """Print available trainer types and exit."""
    print("\nAvailable Trainer Types:")
    print("=" * 70)

    for name, cls in TRAINER_REGISTRY.items():
        doc = cls.__doc__
        if doc:
            # Extract first line of docstring
            description = doc.strip().split('\n')[0]
        else:
            description = "No description available"

        print(f"\n{name}:")
        print(f"  {description}")

    print("\n" + "=" * 70)
    print("\nUse --config <config_file.yaml> to specify a training configuration.")
    print("Example configs are available in the configs/ directory.")


def main() -> None:
    """Main entry point for unified training script."""
    parser = argparse.ArgumentParser(
        description='Unified adversarial patch training script with YAML configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file (required unless --list-trainers)'
    )

    parser.add_argument(
        '--list-trainers',
        action='store_true',
        help='List available trainer types and exit'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate configuration without starting training'
    )

    # Override parameters (optional)
    override_group = parser.add_argument_group('configuration overrides')

    override_group.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size from config'
    )

    override_group.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs from config'
    )

    override_group.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate from config'
    )

    override_group.add_argument(
        '--patch-size',
        type=int,
        help='Override patch size from config'
    )

    override_group.add_argument(
        '--device',
        type=str,
        help='Override device (e.g., "cuda:0", "cpu") from config'
    )

    args = parser.parse_args()

    # Handle --list-trainers
    if args.list_trainers:
        list_available_trainers()
        sys.exit(0)

    # Require config file
    if not args.config:
        parser.error("--config is required (or use --list-trainers)")

    try:
        # Load configuration using Pydantic
        print(f"Loading configuration from: {args.config}")
        config = TrainingConfig.from_yaml(args.config)

        # Apply command-line overrides
        if any([args.batch_size, args.epochs, args.learning_rate,
                args.patch_size, args.device]):
            print("\nApplying command-line overrides...")
            apply_overrides(config, args)

        # Print final configuration
        print("\n" + "=" * 70)
        print("Final Configuration:")
        print("=" * 70)
        print(yaml.dump(config.to_dict(), default_flow_style=False,
                       allow_unicode=True, sort_keys=False))
        print("=" * 70)

        # Validate configuration
        print("\nValidating configuration...")
        config.validate()
        print("✓ Configuration is valid")

        # Exit if validation only
        if args.validate_only:
            print("\n--validate-only specified, exiting without training.")
            sys.exit(0)

        # Create trainer
        print(f"\nCreating trainer: {config.trainer.type}")
        trainer = create_trainer(config)

        # Start training
        print("\nStarting training...\n")
        print("=" * 70)

        # Call trainer.train() - trainers should read config internally
        trainer.train()

        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    except ValueError as e:
        print(f"\nConfiguration error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
