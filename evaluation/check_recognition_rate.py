"""Recognition rate checker for adversarial patches.

This script evaluates the effectiveness of adversarial patches by measuring
detection probabilities with and without the patch applied.

Refactored version using PatchEvaluator class.
"""

import sys
import argparse
from pathlib import Path

from evaluation.patch_evaluator import PatchEvaluator


def main() -> None:
    """Check recognition rate with and without adversarial patch."""
    parser = argparse.ArgumentParser(
        description='Evaluate adversarial patch effectiveness'
    )
    parser.add_argument(
        'mode',
        type=str,
        help='Configuration mode from patch_config.py (e.g., paper_obj)'
    )
    parser.add_argument(
        'patch_path',
        type=str,
        help='Path to adversarial patch image file'
    )
    parser.add_argument(
        '--target-class',
        type=int,
        default=0,
        help='Target class ID to suppress (default: 0 for person in COCO)'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=80,
        help='Total number of classes in dataset (default: 80 for COCO)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=300,
        help='Patch size in pixels (default: 300)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation (default: 8)'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=10,
        help='Number of batches to evaluate (default: 10)'
    )
    parser.add_argument(
        '--no-rotate',
        action='store_true',
        help='Disable rotation augmentation'
    )
    parser.add_argument(
        '--rand-loc',
        action='store_true',
        help='Enable random patch location'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (e.g., cuda:0, cpu). Auto-detected if not specified.'
    )

    args = parser.parse_args()

    # Validate patch file exists
    if not Path(args.patch_path).exists():
        print(f'Error: Patch file not found: {args.patch_path}')
        sys.exit(1)

    # Create evaluator
    print('Initializing patch evaluator...')
    evaluator = PatchEvaluator(
        mode=args.mode,
        target_class=args.target_class,
        num_classes=args.num_classes,
        device=args.device
    )

    # Run comparison
    print(f'\nEvaluating patch: {args.patch_path}')
    print(f'Configuration: {args.mode}')
    print(f'Target class: {args.target_class}')
    print(f'Patch size: {args.patch_size}x{args.patch_size}')
    print(f'Batch size: {args.batch_size}')
    print(f'Number of batches: {args.num_batches}')
    print(f'Rotation: {"disabled" if args.no_rotate else "enabled"}')
    print(f'Random location: {"enabled" if args.rand_loc else "disabled"}')
    print()

    without_patch, with_patch = evaluator.compare_with_without_patch(
        patch_path=args.patch_path,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        max_lab=14,
        num_batches=args.num_batches,
        do_rotate=not args.no_rotate,
        rand_loc=args.rand_loc
    )

    # Print detailed results
    print('\nDetailed Results:')
    print('-' * 60)
    print(f'Without Patch:')
    print(f'  Mean probability: {without_patch["mean_prob"]:.6f}')
    print(f'  Std deviation:    {without_patch["std_prob"]:.6f}')
    print(f'  Samples:          {without_patch["num_samples"]}')
    print()
    print(f'With Patch:')
    print(f'  Mean probability: {with_patch["mean_prob"]:.6f}')
    print(f'  Std deviation:    {with_patch["std_prob"]:.6f}')
    print(f'  Samples:          {with_patch["num_samples"]}')
    print('-' * 60)


if __name__ == '__main__':
    main()
