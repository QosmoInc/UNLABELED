#!/usr/bin/env python3
"""Unified training script for adversarial patch generation.

This script provides a unified interface for training adversarial patches across
different datasets and target classes using YAML configuration files.

Usage:
    # Train with a configuration file
    python train_patch_unified.py --config configs/person_inria.yaml

    # Override configuration with command-line arguments
    python train_patch_unified.py --config configs/person_inria.yaml --batch-size 8 --epochs 500

    # List available trainer types
    python train_patch_unified.py --list-trainers

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
from typing import Any, Dict, Optional

import yaml

import trainers


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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Check required top-level keys
    required_keys = ['trainer']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Check trainer configuration
    if 'type' not in config['trainer']:
        raise ValueError("Missing 'type' in trainer configuration")

    trainer_type = config['trainer']['type']
    if trainer_type not in TRAINER_REGISTRY:
        available = ', '.join(TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Available types: {available}"
        )

    # Check mode is specified
    if 'mode' not in config['trainer']:
        raise ValueError("Missing 'mode' in trainer configuration")


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply command-line argument overrides to configuration.

    Args:
        config: Configuration dictionary to modify
        args: Parsed command-line arguments
    """
    # Override training parameters
    if args.batch_size is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['batch_size'] = args.batch_size

    if args.epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = args.epochs

    if args.learning_rate is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['learning_rate'] = args.learning_rate

    # Override patch parameters
    if args.patch_size is not None:
        if 'patch' not in config:
            config['patch'] = {}
        config['patch']['size'] = args.patch_size

    # Override device
    if args.device is not None:
        if 'trainer' not in config:
            config['trainer'] = {}
        config['trainer']['device'] = args.device


def create_trainer(config: Dict[str, Any]) -> trainers.BasePatchTrainer:
    """Create trainer instance from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Trainer instance

    Raises:
        ValueError: If trainer type is unknown or configuration is invalid
    """
    trainer_config = config['trainer']
    trainer_type = trainer_config['type']
    mode = trainer_config['mode']
    device = trainer_config.get('device', None)

    # Get trainer class
    trainer_class = TRAINER_REGISTRY[trainer_type]

    # Create trainer instance
    # MultiClassPatchTrainer has additional parameters
    if trainer_type == 'MultiClassPatchTrainer':
        targets = config.get('targets', {})
        suppress_class = targets.get('suppress', {}).get('class_id', 0)
        enhance_class = targets.get('enhance', {}).get('class_id', 21)

        trainer = trainer_class(
            mode=mode,
            suppress_class_id=suppress_class,
            enhance_class_id=enhance_class,
            device=device
        )
    else:
        trainer = trainer_class(mode=mode, device=device)

    return trainer


def list_available_trainers() -> None:
    """Print available trainer types and exit."""
    print("\nAvailable Trainer Types:")
    print("=" * 60)

    for name, cls in TRAINER_REGISTRY.items():
        doc = cls.__doc__
        if doc:
            # Extract first line of docstring
            description = doc.strip().split('\n')[0]
        else:
            description = "No description available"

        print(f"\n{name}:")
        print(f"  {description}")

    print("\n" + "=" * 60)
    print("\nUse --config <config_file.yaml> to specify a training configuration.")
    print("Example configs are available in the configs/ directory.")


def main() -> None:
    """Main entry point for unified training script."""
    parser = argparse.ArgumentParser(
        description='Unified adversarial patch training script',
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
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        # Validate configuration
        validate_config(config)

        # Apply command-line overrides
        apply_overrides(config, args)

        # Print final configuration
        print("\nFinal Configuration:")
        print("=" * 60)
        print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        print("=" * 60)

        # Create trainer
        print(f"\nCreating trainer: {config['trainer']['type']}")
        trainer = create_trainer(config)

        # Start training
        print("\nStarting training...\n")

        # Special handling for MultiClassPatchTrainer
        if isinstance(trainer, trainers.MultiClassPatchTrainer):
            patch_config = config.get('patch', {})
            style_image = patch_config.get('style_image', 'imgs/210825_ダギング_6.jpg')
            content_image = patch_config.get('content_image', 'imgs/bear2.jpg')
            trainer.train(style_image_path=style_image, content_image_path=content_image)
        else:
            trainer.train()

        print("\nTraining completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    except (ValueError, yaml.YAMLError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
