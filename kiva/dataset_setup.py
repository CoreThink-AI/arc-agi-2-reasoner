"""
KiVA Dataset Setup

Run this script to check your KiVA dataset status and get setup instructions.

    python -m kiva.dataset_setup

Two options to get the data:

OPTION A — Clone the repo (data may already be included):
    git clone https://github.com/ey242/kiva  <path/to/kiva-repo>

OPTION B — Generate the images yourself using their PyTorch scripts:
    cd <path/to/kiva-repo>
    python transformation/pytorch_transformations_kiva.py \
        --input_dir "untransformed objects" \
        --output_dir "transformed objects" \
        --transformation Colour \
        --num_trials 10
    # Repeat for each concept: Counting, Resize, Reflect, 2DRotation

After either option, set the env var in your .env file:
    KIVA_DATA_DIR=/absolute/path/to/kiva-repo
"""

import os
import sys
from pathlib import Path


REQUIRED_CONCEPTS = [
    ("Counting", ["+1", "+2", "-1", "-2"]),
    ("Resize",   ["0.5XY", "2XY"]),
    ("Colour",   ["Red", "Green", "Blue"]),
    ("Reflect",  ["X", "Y"]),
    ("2DRotation", ["+90", "-90", "180"]),
]


def check_dataset(root: Path) -> bool:
    data_dir = root / "transformed objects" / "KiVA"
    if not data_dir.exists():
        print(f"  [MISSING] {data_dir}")
        return False

    all_ok = True
    for concept, params in REQUIRED_CONCEPTS:
        for param in params:
            # Try both underscore and space variants
            found = False
            for sep in ["_", " "]:
                d = data_dir / f"{concept}{sep}{param}"
                if d.exists():
                    n_pngs = len(list(d.glob("*.png")))
                    print(f"  [OK]      {concept}/{param}  ({n_pngs} images)")
                    found = True
                    break
            if not found:
                print(f"  [MISSING] {concept}/{param}")
                all_ok = False

    return all_ok


def main():
    kiva_root_str = os.getenv("KIVA_DATA_DIR", "")
    if not kiva_root_str:
        print("KIVA_DATA_DIR is not set in your environment.\n")
        print("Steps to set up the KiVA dataset:")
        print()
        print("  1. Clone the KiVA repo:")
        print("       git clone https://github.com/ey242/kiva  /path/to/kiva")
        print()
        print("  2. Add to your .env file:")
        print("       KIVA_DATA_DIR=/path/to/kiva")
        print()
        print("  3. If 'transformed objects/KiVA/' is empty, generate images:")
        print("       cd /path/to/kiva")
        print("       python transformation/pytorch_transformations_kiva.py \\")
        print("           --input_dir 'untransformed objects' \\")
        print("           --output_dir 'transformed objects' \\")
        print("           --transformation Colour --num_trials 10")
        print("       # (repeat for Counting, Resize, Reflect, 2DRotation)")
        sys.exit(1)

    root = Path(kiva_root_str)
    print(f"Checking KiVA dataset at: {root}\n")

    if not root.exists():
        print(f"ERROR: Path does not exist: {root}")
        sys.exit(1)

    ok = check_dataset(root)

    if ok:
        print("\nDataset looks complete. You are ready to run the KiVA pipeline.")
    else:
        print("\nSome concept directories are missing.")
        print("See the instructions above (OPTION B) to generate missing data.")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
