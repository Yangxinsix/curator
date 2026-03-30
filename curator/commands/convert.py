import argparse
from pathlib import Path
from typing import List, Optional

from .common import argcomplete, prepare_cli_environment


def _convert_parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert original MACE checkpoints to Curator format or upgrade Curator checkpoints",
        fromfile_prefix_chars="+",
    )
    parser.add_argument("ckpt_path", metavar="INPUT_FILE", type=str, help="Path to a MACE or Curator checkpoint to convert")
    parser.add_argument("-o", "--output", type=str, help="Output path for the converted checkpoint (default: alongside input with _converted suffix)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading the checkpoint; defaults to CPU so conversion works without GPUs")
    parser.add_argument("-u", "--update", action="store_true", help="Update an old CURATOR checkpoint by rebuilding the stored model.")
    parser.add_argument("--e3nn-to-cueq", action="store_true", help="Convert a Curator e3nn model checkpoint to cuequivariance backend.")
    parser.add_argument("--cueq-to-e3nn", action="store_true", help="Convert a Curator cuequivariance model checkpoint back to e3nn backend.")
    parser.add_argument("--mace-to-curator", action="store_true", help="Convert an original MACE checkpoint to a Curator MACE checkpoint.")
    parser.add_argument("--curator-to-mace", action="store_true", help="Convert a Curator MACE checkpoint back to an original MACE checkpoint.")
    if argcomplete:
        argcomplete.autocomplete(parser)
    return parser.parse_args(argv)


def convert_main(argv: Optional[List[str]] = None):
    prepare_cli_environment()
    args = _convert_parse_args(argv)
    from ..model.checkpoint_upgrade import upgrade_checkpoint
    from ..model.conversion import (
        convert_cueq_to_e3nn,
        convert_curator_to_mace,
        convert_e3nn_to_cueq,
        convert_mace_to_curator,
    )
    from ..utils import load_models

    device = args.device
    target = None

    if args.update:
        target = upgrade_checkpoint(ckpt_path=args.ckpt_path, output_path=args.output, device=device)
    elif args.e3nn_to_cueq or args.cueq_to_e3nn:
        import torch

        from ..layer._cuequivariance_wrapper import IS_CUET_AVAILABLE

        if args.cueq_to_e3nn and (not torch.cuda.is_available() or not IS_CUET_AVAILABLE):
            raise RuntimeError(
                "Converting from cueq to e3nn requires cuequivariance with CUDA. "
                "Please run on a CUDA-enabled environment with cuequivariance installed."
            )

        try:
            models = load_models(args.ckpt_path, device=torch.device(device), load_compiled=False)
        except PermissionError as exc:
            raise RuntimeError(
                "Failed to load cueq checkpoint due to permission/CUDA issues. "
                "cuequivariance typically requires CUDA; please run on a CUDA-enabled setup "
                "with cuequivariance installed."
            ) from exc
        except Exception as exc:
            if args.cueq_to_e3nn:
                raise RuntimeError(
                    "Failed to load cueq checkpoint. Ensure cuequivariance is installed and CUDA is available."
                ) from exc
            raise
        if len(models) != 1:
            raise ValueError("Cueq/e3nn conversion supports single-model checkpoints only.")
        model = models[0]

        if args.e3nn_to_cueq and args.cueq_to_e3nn:
            raise ValueError("Choose only one of --e3nn-to-cueq or --cueq-to-e3nn.")
        if args.e3nn_to_cueq:
            converted = convert_e3nn_to_cueq(model)
            suffix = "_cueq"
        else:
            converted = convert_cueq_to_e3nn(model)
            suffix = "_e3nn"

        ckpt_path = Path(args.ckpt_path)
        output_path = args.output or ckpt_path.with_name(f"{ckpt_path.stem}{suffix}{ckpt_path.suffix}")
        torch.save(converted, output_path)
        target = output_path
    elif args.mace_to_curator or args.curator_to_mace:
        import torch

        ckpt_path = Path(args.ckpt_path)
        output_path = args.output
        if output_path is None:
            suffix = "_mace" if args.curator_to_mace else "_converted"
            output_path = ckpt_path.with_name(f"{ckpt_path.stem}{suffix}{ckpt_path.suffix}")
        if args.curator_to_mace:
            target = convert_curator_to_mace(curator_path=ckpt_path, output_path=output_path, device=torch.device(device))
        else:
            target = convert_mace_to_curator(mace_path=ckpt_path, output_path=output_path, device=torch.device(device))
    else:
        import torch

        ckpt_path = Path(args.ckpt_path)
        output_path = args.output or ckpt_path.with_name(f"{ckpt_path.stem}_converted{ckpt_path.suffix}")
        target = convert_mace_to_curator(mace_path=ckpt_path, output_path=output_path, device=torch.device(device))

    print(f"Converted checkpoint saved to {target}")
    return target
