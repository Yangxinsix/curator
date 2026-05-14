#!/usr/bin/env python
"""
Script to create LAMMPS MLIAP model files from Curator models.

Usage:
    # For standard Curator model (no QEQ)
    python -m curator.simulate.create_lammps_model model.ckpt --output model-lammps.pt

    # For Curator-QEQ model (with charge equilibration)
    python -m curator.simulate.create_lammps_model model.ckpt --qeq --output model-qeq-lammps.pt

After creating the model file, use in LAMMPS:
    
    # In your LAMMPS input file:
    pair_style mliap unified model-lammps.pt
    pair_coeff * * H C N O  # element types in same order as model

    # For QEQ with LAMMPS kspace:
    pair_style hybrid/overlay mliap unified model-qeq-lammps.pt coul/long 10.0
    pair_coeff * * mliap H C N O
    pair_coeff * * coul/long
    kspace_style ewald 1e-6
    kspace_modify gewald 0.4  # Must match model's alpha value!

Running with multiple GPUs:
    mpirun -np 2 lmp -k on g 2 -sf kk -pk kokkos newton on neigh half -in input.in
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_curator_model(model_path: str, device: str = 'cpu'):
    """Load a Curator model from checkpoint."""
    import sys
    # Add curator package to path if running from interface directory
    script_dir = Path(__file__).parent.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    from curator.model.base import LitNNP, NeuralNetworkPotential
    
    path = Path(model_path)
    
    if path.suffix == '.ckpt':
        # PyTorch Lightning checkpoint
        logger.info(f"Loading Lightning checkpoint: {model_path}")
        
        # Load checkpoint manually to handle various formats
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if checkpoint has the full model object directly
        if 'model' in ckpt and isinstance(ckpt['model'], NeuralNetworkPotential):
            logger.info("Found NeuralNetworkPotential directly in checkpoint")
            model = ckpt['model']
        else:
            # Try LitNNP.load_from_checkpoint
            try:
                lit_model = LitNNP.load_from_checkpoint(model_path, map_location=device)
                model = lit_model.model
            except Exception as e:
                logger.warning(f"Could not load with LitNNP: {e}")
                raise ValueError(f"Cannot load model from checkpoint: {e}")
                
    elif path.suffix == '.pt':
        # Direct torch save
        logger.info(f"Loading torch model: {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(model, LitNNP):
            model = model.model
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")
    
    return model


def create_lammps_model(
    model_path: str,
    output_path: Optional[str] = None,
    qeq: bool = False,
    element_types: Optional[List[str]] = None,
    dtype: str = 'float64',
    device: str = 'cpu',
    use_lammps_kspace: bool = True,
    total_charge: float = 0.0,
) -> str:
    """
    Create a LAMMPS MLIAP model file from a Curator model.
    
    Args:
        model_path: Path to Curator model checkpoint (.ckpt or .pt)
        output_path: Output path for LAMMPS model file (default: model-lammps.pt)
        qeq: Whether to use QEQ interface (for charge equilibration models)
        element_types: List of element symbols (default: inferred from model)
        dtype: Data type ('float32' or 'float64')
        device: Device to use for model ('cpu' or 'cuda')
        use_lammps_kspace: For QEQ, whether LAMMPS computes kspace (recommended)
        total_charge: For QEQ, total system charge constraint
    
    Returns:
        Path to the created LAMMPS model file
    """
    # Note: Do NOT change the default dtype here
    # The model was trained with a specific dtype (usually float32)
    # LAMMPS passes float64 data, but we convert it to match model's dtype in _prepare_batch
    # Using model's original dtype is important for cuequivariance compatibility
    
    # Load model
    model = load_curator_model(model_path, device)
    
    # Determine model's dtype from its parameters
    model_dtype = next(model.parameters()).dtype
    logger.info(f"Model dtype: {model_dtype}")
    
    # Configure model for LAMMPS mode:
    # - LAMMPS provides edge_diff (rij) directly, no need to compute from positions
    # - Forces are computed as pair forces (fij), not atomic forces
    from curator.layer._pairwise_distance import PairwiseDistance
    for module in model.modules():
        if isinstance(module, PairwiseDistance):
            module.compute_distance_from_R = False
            logger.info(f"Set PairwiseDistance.compute_distance_from_R = False for LAMMPS mode")
    
    # Get element types from model if not specified
    if element_types is None:
        if hasattr(model, 'representation') and hasattr(model.representation, 'species'):
            element_types = model.representation.species
            logger.info(f"Inferred element types from model: {element_types}")
        else:
            raise ValueError("Could not infer element types from model. Please specify --elements")
    
    # Create LAMMPS interface
    if qeq:
        from curator.simulate.lammps_mliap_interface import LAMMPS_MLIAP_QEQ
        logger.info("Creating LAMMPS_MLIAP_QEQ interface...")
        lammps_model = LAMMPS_MLIAP_QEQ(
            model=model,
            element_types=element_types,
            use_lammps_kspace=use_lammps_kspace,
            total_charge=total_charge,
        )
        default_suffix = "-qeq-lammps.pt"
    else:
        from curator.simulate.lammps_mliap_interface import LAMMPS_MLIAP
        logger.info("Creating LAMMPS_MLIAP interface...")
        lammps_model = LAMMPS_MLIAP(
            model=model,
            element_types=element_types,
        )
        default_suffix = "-lammps.pt"
    
    # Determine output path
    if output_path is None:
        output_path = str(Path(model_path).with_suffix('')) + default_suffix
    
    # Save model
    logger.info(f"Saving LAMMPS model to: {output_path}")
    torch.save(lammps_model, output_path)
    
    # Print usage instructions
    print("\n" + "="*70)
    print("LAMMPS model created successfully!")
    print("="*70)
    print(f"\nModel file: {output_path}")
    print(f"Element types: {' '.join(element_types)}")
    
    if qeq:
        print(f"\nFor QEQ model with LAMMPS kspace, use in your LAMMPS input file:")
        print(f"""
# LAMMPS input example for Curator-QEQ
units metal
atom_style charge

# Read structure
read_data your_structure.data

# Hybrid pair style: ML short-range + Coulomb long-range
pair_style hybrid/overlay mliap unified {output_path} coul/long 10.0
pair_coeff * * mliap {' '.join(element_types)}
pair_coeff * * coul/long

# Ewald summation for long-range Coulomb
# IMPORTANT: gewald (alpha) must match model training value!
kspace_style ewald 1e-6
kspace_modify gewald 0.4

# Run with Kokkos for GPU acceleration:
# mpirun -np 2 lmp -k on g 2 -sf kk -pk kokkos newton on neigh half -in input.in
""")
    else:
        print(f"\nFor standard model, use in your LAMMPS input file:")
        print(f"""
# LAMMPS input example for Curator
units metal
atom_style atomic

# Read structure
read_data your_structure.data

pair_style mliap unified {output_path}
pair_coeff * * {' '.join(element_types)}

# Run with Kokkos for GPU acceleration:
# mpirun -np 2 lmp -k on g 2 -sf kk -pk kokkos newton on neigh half -in input.in
""")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create LAMMPS MLIAP model file from Curator model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to Curator model checkpoint (.ckpt or .pt)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for LAMMPS model file"
    )
    parser.add_argument(
        "--qeq",
        action="store_true",
        help="Use QEQ interface for charge equilibration models"
    )
    parser.add_argument(
        "--elements", "-e",
        type=str,
        nargs="+",
        default=None,
        help="Element symbols (e.g., H C N O). If not specified, inferred from model."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="Data type for model (default: float64)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model loading (default: cpu)"
    )
    parser.add_argument(
        "--no-lammps-kspace",
        action="store_true",
        help="For QEQ: compute Ewald in Python instead of LAMMPS (not recommended)"
    )
    parser.add_argument(
        "--total-charge",
        type=float,
        default=0.0,
        help="For QEQ: total system charge constraint (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    try:
        create_lammps_model(
            model_path=args.model_path,
            output_path=args.output,
            qeq=args.qeq,
            element_types=args.elements,
            dtype=args.dtype,
            device=args.device,
            use_lammps_kspace=not args.no_lammps_kspace,
            total_charge=args.total_charge,
        )
    except Exception as e:
        logger.error(f"Failed to create LAMMPS model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
