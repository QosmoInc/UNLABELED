#!/usr/bin/env python3
"""
Check the best (lowest/highest recognition rate) patch image in a directory.
Image filename format: {epoch}_{recognition_rate}.png
Example: 99_0.755788.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional


def parse_filename(filename: str) -> Optional[Tuple[int, float]]:
    """
    Parse image filename to extract epoch and recognition rate.

    Args:
        filename: Image filename (e.g., "99_0.755788.png")

    Returns:
        Tuple of (epoch, recognition_rate) or None if parsing fails
    """
    try:
        # Remove extension
        name_without_ext = Path(filename).stem

        # Split by underscore
        parts = name_without_ext.split('_')
        if len(parts) != 2:
            return None

        epoch = int(parts[0])
        recognition_rate = float(parts[1])

        return (epoch, recognition_rate)
    except (ValueError, IndexError):
        return None


def find_best_patch(directory: str, find_lowest: bool = True, exclude_epoch_zero: bool = True) -> Optional[Tuple[str, int, float]]:
    """
    Find the patch image with the lowest or highest recognition rate.

    Args:
        directory: Directory path containing patch images
        find_lowest: If True, find lowest rate; if False, find highest rate
        exclude_epoch_zero: If True, exclude epoch 0 from results

    Returns:
        Tuple of (filename, epoch, recognition_rate) or None if no valid files found
    """
    image_extensions = {'.png', '.jpg', '.jpeg'}
    best_file = None
    best_epoch = None
    best_rate = None

    for filename in os.listdir(directory):
        # Skip files starting with "best_"
        if filename.lower().startswith('best_'):
            continue

        # Check if it's an image file
        if Path(filename).suffix.lower() not in image_extensions:
            continue

        # Parse filename
        parsed = parse_filename(filename)
        if parsed is None:
            continue

        epoch, rate = parsed

        # Skip epoch 0 if requested
        if exclude_epoch_zero and epoch == 0:
            continue

        # Update best file
        if best_rate is None:
            best_file = filename
            best_epoch = epoch
            best_rate = rate
        elif find_lowest and rate < best_rate:
            best_file = filename
            best_epoch = epoch
            best_rate = rate
        elif not find_lowest and rate > best_rate:
            best_file = filename
            best_epoch = epoch
            best_rate = rate

    if best_file is None:
        return None

    return (best_file, best_epoch, best_rate)


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Find the best adversarial patch with lowest recognition rate"
    )
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default="imgs",
        help="Directory containing patch images (default: imgs)"
    )
    parser.add_argument(
        "--include-epoch-zero",
        action="store_true",
        help="Include epoch 0 in the search"
    )

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.directory):
        print(f"エラー: ディレクトリ '{args.directory}' が見つかりません")
        sys.exit(1)

    # Find lowest recognition rate (best for adversarial patch)
    print(f"ディレクトリ: {args.directory}")
    print("-" * 60)

    exclude_epoch_zero = not args.include_epoch_zero
    lowest = find_best_patch(args.directory, find_lowest=True, exclude_epoch_zero=exclude_epoch_zero)
    if lowest:
        filename, epoch, rate = lowest
        print(f"最低認識率 (ベスト):")
        print(f"  ファイル名: {filename}")
        print(f"  エポック: {epoch}")
        print(f"  認識率: {rate:.6f}")
    else:
        print("最低認識率: 有効な画像ファイルが見つかりません")


if __name__ == "__main__":
    main()
