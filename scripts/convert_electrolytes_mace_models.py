import argparse
from pathlib import Path

import torch
import torch.serialization as torch_serialization

from curator.utils import create_model_from_mace


def _load_mace_model(path: Path, device: torch.device) -> torch.nn.Module:
    torch_serialization.add_safe_globals([slice])
    try:
        from mace.modules.models import ScaleShiftMACE

        torch_serialization.add_safe_globals([ScaleShiftMACE])
    except Exception:
        pass

    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        model = obj.get("model")
        if model is not None:
            return model
    raise TypeError(f"Unsupported MACE checkpoint format at {path}")


def _parse_head_overrides(items):
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid head override '{item}'. Use filename=head_name.")
        name, head_name = item.split("=", 1)
        mapping[name] = head_name
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all electrolytes MACE checkpoints to Curator models."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("electrolytes/mace-model"),
        help="Directory containing the .model checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("electrolytes/curator-model"),
        help="Destination directory for Curator models.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string used for loading (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--head",
        action="append",
        default=[],
        help="Override head selection per file, e.g. name.model=pt_head.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    head_overrides = _parse_head_overrides(args.head)
    model_paths = sorted(input_dir.glob("*.model"))
    if not model_paths:
        raise FileNotFoundError(f"No .model files found in {input_dir}")

    for path in model_paths:
        mace_model = _load_mace_model(path, device)
        head = head_overrides.get(path.name)
        heads = list(getattr(mace_model, "heads", []))
        if head is None and len(heads) > 1 and "pt_head" in heads and "default_head_only" not in path.name:
            head = "pt_head"

        curator_model = create_model_from_mace(mace_model, head=head)
        out_path = output_dir / f"{path.stem}.curator.pth"
        torch.save(curator_model, out_path)
        head_note = head if head is not None else "auto"
        print(f"{path.name} -> {out_path} (head={head_note})")


if __name__ == "__main__":
    main()
