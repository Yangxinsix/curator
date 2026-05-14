#!/usr/bin/env python
"""
Test script for Curator LAMMPS MLIAP interface.

This script tests the interface components without requiring LAMMPS.
"""

import sys
import torch

def test_imports():
    """Test that all components can be imported."""
    print("=" * 60)
    print("Test 1: Imports")
    print("=" * 60)
    
    try:
        from curator.simulate import (
            LAMMPS_MLIAP, 
            LAMMPS_MLIAP_QEQ, 
            prepare_model_for_qeq_inference,
            create_lammps_model,
        )
        print("✓ All LAMMPS MLIAP components imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from curator.data import properties
        assert hasattr(properties, 'lammps_data'), "Missing lammps_data property"
        assert hasattr(properties, 'n_local'), "Missing n_local property"
        assert hasattr(properties, 'n_ghost'), "Missing n_ghost property"
        print("✓ LAMMPS properties defined in properties.py")
    except Exception as e:
        print(f"✗ Properties error: {e}")
        return False
    
    try:
        from curator.layer._interaction import LammpsMessagePassing, Interaction
        print("✓ LammpsMessagePassing class available")
    except ImportError as e:
        print(f"✗ LammpsMessagePassing import error: {e}")
        return False
    
    return True


def test_interaction_layer():
    """Test that interaction layer has LAMMPS support."""
    print("\n" + "=" * 60)
    print("Test 2: Interaction Layer LAMMPS Support")
    print("=" * 60)
    
    from curator.layer._mace_interaction import (
        RealAgnosticInteractionBlock,
        RealAgnosticResidualInteractionBlock,
    )
    from curator.layer._interaction import Interaction
    import inspect
    
    # Check that exchange_info method exists
    assert hasattr(Interaction, 'exchange_info'), "Missing exchange_info method"
    assert hasattr(Interaction, 'truncate_ghost'), "Missing truncate_ghost method"
    print("✓ Interaction base class has exchange_info and truncate_ghost methods")
    
    # Check forward signature has LAMMPS parameters
    for cls in [RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock]:
        sig = inspect.signature(cls.forward)
        params = list(sig.parameters.keys())
        assert 'lammps_data' in params, f"{cls.__name__} missing lammps_data param"
        assert 'n_local' in params, f"{cls.__name__} missing n_local param"
        assert 'n_ghost' in params, f"{cls.__name__} missing n_ghost param"
        print(f"✓ {cls.__name__}.forward() has LAMMPS parameters")
    
    return True


def test_mace_forward():
    """Test that MACE model forward passes LAMMPS data."""
    print("\n" + "=" * 60)
    print("Test 3: MACE Model LAMMPS Data Passing")
    print("=" * 60)
    
    import inspect
    from curator.model.mace import MACE
    
    # Check MACE.forward source code
    source = inspect.getsource(MACE.forward)
    
    checks = [
        ('lammps_data = data.get', "lammps_data extraction"),
        ('n_local = data.get', "n_local extraction"),
        ('n_ghost = data.get', "n_ghost extraction"),
        ('lammps_data=lammps_data', "lammps_data passing to interaction"),
        ('is_first_layer=is_first_layer', "is_first_layer passing"),
    ]
    
    for pattern, description in checks:
        if pattern in source:
            print(f"✓ MACE.forward contains: {description}")
        else:
            print(f"✗ MACE.forward missing: {description}")
            return False
    
    return True


def test_model_loading():
    """Test that we can create a MACE model."""
    print("\n" + "=" * 60)
    print("Test 4: Model Creation")
    print("=" * 60)
    
    from curator.model.mace import MACE
    from curator.data import properties
    
    # Create a minimal MACE model
    model = MACE(
        cutoff=5.0,
        num_interactions=2,
        correlation=3,
        species=['H', 'C', 'N', 'O'],
        hidden_irreps='32x0e + 32x1o',
        avg_num_neighbors=10.0,
    )
    
    print(f"✓ Created MACE model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward with mock LAMMPS data
    batch = {
        properties.atomic_numbers: torch.tensor([1, 6, 7, 8]),
        properties.edge_idx: torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 3, 0, 3]]).T,
        properties.edge_diff: torch.randn(6, 3),
        properties.edge_dist: torch.rand(6) * 5.0,
        properties.n_atoms: torch.tensor([4]),
        # LAMMPS data (None for non-LAMMPS runs)
        properties.lammps_data: None,
        properties.n_local: None,
        properties.n_ghost: None,
    }
    
    with torch.no_grad():
        output = model(batch)
    
    print(f"✓ Forward pass successful, output keys: {list(output.keys())}")
    
    return True


def test_lammps_interface_creation():
    """Test creating LAMMPS MLIAP interface."""
    print("\n" + "=" * 60)
    print("Test 5: LAMMPS MLIAP Interface Creation")
    print("=" * 60)
    
    from curator.model.mace import MACE
    from curator.model.base import NeuralNetworkPotential
    from curator.layer import AtomwiseNN, GradientOutput, GlobalRescaleShift
    from curator.simulate import LAMMPS_MLIAP, LAMMPS_MLIAP_QEQ
    
    # Create a minimal MACE representation
    mace = MACE(
        cutoff=5.0,
        num_interactions=2,
        correlation=3,
        species=['H', 'C', 'N', 'O'],
        hidden_irreps='32x0e + 32x1o',
        avg_num_neighbors=10.0,
    )
    
    # Create NNP wrapper
    model = NeuralNetworkPotential(
        representation=mace,
        output_modules=[
            GradientOutput(compute_forces=True),
            GlobalRescaleShift(scale=1.0, shift=0.0),
        ],
    )
    
    # Test LAMMPS_MLIAP
    try:
        lammps_mliap = LAMMPS_MLIAP(model)
        print(f"✓ Created LAMMPS_MLIAP interface")
        print(f"  - element_types: {lammps_mliap.element_types}")
        print(f"  - rcutfac: {lammps_mliap.rcutfac}")
    except Exception as e:
        print(f"✗ LAMMPS_MLIAP creation failed: {e}")
        return False
    
    # Test LAMMPS_MLIAP_QEQ
    try:
        lammps_qeq = LAMMPS_MLIAP_QEQ(model, use_lammps_kspace=True)
        print(f"✓ Created LAMMPS_MLIAP_QEQ interface")
        print(f"  - use_lammps_kspace: {lammps_qeq.use_lammps_kspace}")
    except Exception as e:
        print(f"✗ LAMMPS_MLIAP_QEQ creation failed: {e}")
        return False
    
    return True


def test_serialization():
    """Test model serialization."""
    print("\n" + "=" * 60)
    print("Test 6: Model Serialization")
    print("=" * 60)
    
    import tempfile
    import os
    from curator.model.mace import MACE
    from curator.model.base import NeuralNetworkPotential
    from curator.layer import GradientOutput, GlobalRescaleShift
    from curator.simulate import LAMMPS_MLIAP
    
    # Create model
    mace = MACE(
        cutoff=5.0,
        num_interactions=2,
        correlation=3,
        species=['H', 'C'],
        hidden_irreps='16x0e',
        avg_num_neighbors=5.0,
    )
    
    model = NeuralNetworkPotential(
        representation=mace,
        output_modules=[
            GradientOutput(compute_forces=True),
            GlobalRescaleShift(scale=1.0, shift=0.0),
        ],
    )
    
    lammps_mliap = LAMMPS_MLIAP(model)
    
    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    try:
        torch.save(lammps_mliap, temp_path)
        file_size = os.path.getsize(temp_path) / 1024
        print(f"✓ Saved LAMMPS model to {temp_path} ({file_size:.1f} KB)")
        
        loaded = torch.load(temp_path, weights_only=False)
        print(f"✓ Loaded LAMMPS model successfully")
        print(f"  - type: {type(loaded).__name__}")
        print(f"  - element_types: {loaded.element_types}")
    except Exception as e:
        print(f"✗ Serialization failed: {e}")
        return False
    finally:
        os.unlink(temp_path)
    
    return True


def main():
    print("Curator LAMMPS MLIAP Interface Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Interaction Layer", test_interaction_layer),
        ("MACE Forward", test_mace_forward),
        ("Model Loading", test_model_loading),
        ("LAMMPS Interface", test_lammps_interface_creation),
        ("Serialization", test_serialization),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
