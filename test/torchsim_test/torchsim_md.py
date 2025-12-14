"""
Minimal TorchSim MD example using a CURATOR checkpoint.

Prereqs:
    - torch-sim installed (importable as `torch_sim`)
    - CURATOR checkpoint at notebooks/best_model.ckpt
    - Initial structure: first frame of curator/example/LiFePO4.traj

Run:
    python interface/examples/torchsim_md.py
"""

import torch
from ase.io import Trajectory
from curator.interface import CuratorTorchSimAdapter
from curator.simulate.engines.torchsim import TorchSimEngine
from curator.simulate.callbacks.torchsim import TorchSimThermoLogger
from curator.simulate.core.simulator import Simulator
import time


def main():
    atoms = Trajectory("..//LiFePO4.traj")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = CuratorTorchSimAdapter(
        "../best_model.ckpt",
        # match paths relative to repo root when running this script
        compute_neighbor_list=True,
        cutoff=None,  # infer from model if available
        detach=True,
        device=device,
        load_compiled=False,
    )

    engine = TorchSimEngine(model=adapter, integrator="nve", temperature=300.0, timestep=1e-3)
    thermo = TorchSimThermoLogger(interval=10, variables=["step", "epot", "natoms"])

    start = time.time()
    sim = Simulator(atoms, engine, callbacks=[thermo], start_index=None, logger=None)
    sim.run(steps=100)

    state = engine.state
    print("Final potential energy (eV):", state.energy)
    print("Final forces shape:", state.forces.shape)
    print(f"Time for this simulation: {time.time() - start} s")


if __name__ == "__main__":
    main()
