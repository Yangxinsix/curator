#!/usr/bin/env python3
import curator.cli as cli
import argparse

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Script for deploy curator models", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "model_path",
        metavar="INPUT_FILE",
        type=str,
        nargs="+",
        help="Path to model that to be compiled",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="compiled_model.pt",
        help="Path to save compiled model",
    )
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        help="Load trained weights while initialize the model",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Configuration file that defines model parameters",
    )
    parser.add_argument(
        "--lammps",
        action="store_true",
        help="Deploy model for LAMMPS MLIAP package",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        help="Elements that used for LAMMPS MLIAP model",
    )

    return parser.parse_args(arg_list)

if __name__ == "__main__":
    args = get_arguments()
    cli.deploy(
        model_path=args.model_path,
        target_path=args.target_path,
        load_weights_only=args.load_weights_only,
        lammps_mliap=args.lammps,
        element_types=args.elements,
    )