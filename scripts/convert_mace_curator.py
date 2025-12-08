import argparse
from pathlib import Path
import torch

from curator.utils import convert_mace_to_curator, convert_curator_to_mace


def main():
    parser = argparse.ArgumentParser(description="Convert between MACE and Curator checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_curator = subparsers.add_parser("to_curator", help="Convert a MACE checkpoint to Curator format.")
    to_curator.add_argument("mace_checkpoint", type=Path, help="Path to the MACE checkpoint (.pt/.pth).")
    to_curator.add_argument("output", type=Path, help="Destination path for the Curator model.")
    to_curator.add_argument("--foundation", action="store_true", help="Use foundation-style first interaction block.")

    to_mace = subparsers.add_parser("to_mace", help="Convert a Curator checkpoint to MACE format.")
    to_mace.add_argument("curator_checkpoint", type=Path, help="Path to the Curator checkpoint (.pt/.ckpt).")
    to_mace.add_argument("output", type=Path, help="Destination path for the MACE model.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "to_curator":
        dest = convert_mace_to_curator(args.mace_checkpoint, args.output, foundation=args.foundation, device=device)
        print(f"Saved Curator model to {dest}")
    elif args.command == "to_mace":
        dest = convert_curator_to_mace(args.curator_checkpoint, args.output, device=device)
        print(f"Saved MACE model to {dest}")


if __name__ == "__main__":
    main()
