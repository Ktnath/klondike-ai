from __future__ import annotations

"""Command line interface for dataset validation."""

import argparse
import sys

from core.validate_dataset import validate_npz_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NPZ dataset")
    parser.add_argument("--path", required=True, help="Path to dataset .npz file")
    parser.add_argument("--use_intentions", action="store_true", help="Expect intention labels")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()

    valid, _ = validate_npz_dataset(args.path, args.use_intentions, verbose=not args.no_verbose)

    if args.use_intentions:
        msg = "Dataset valide avec intentions" if valid else "Dataset non valide avec intentions"
    else:
        msg = "Dataset valide sans intentions" if valid else "Dataset non valide"

    emoji = "✅" if valid else "❌"
    print(f"{emoji} {msg}")
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
