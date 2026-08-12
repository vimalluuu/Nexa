"""
Create and maintain Nexa's dataset directory structure.

This script is safe to run repeatedly. It creates missing folders and metadata
templates but never deletes data and never modifies existing metadata files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from nexa.data import ensure_dataset_directory, ensure_dataset_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensure Nexa dataset pipeline folders exist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--dataset-name",
        help="Optional source dataset directory to create under raw/cleaned/deduplicated.",
    )
    parser.add_argument(
        "--stage",
        choices=("raw", "cleaned", "deduplicated"),
        default="raw",
        help="Dataset stage used with --dataset-name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories = ensure_dataset_layout(args.data_root)
    print(f"Ensured {len(directories)} dataset pipeline directories under {args.data_root}.")

    if args.dataset_name:
        dataset_dir = ensure_dataset_directory(args.dataset_name, args.stage, args.data_root)
        print(f"Ensured dataset directory: {dataset_dir}")


if __name__ == "__main__":
    main()
