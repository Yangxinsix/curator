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
from torch_sim import integrate
from torch_sim.integrators import nve
from torch_sim.trajectory import TrajectoryReporter
import time


def main():
    atoms = Trajectory("../LiFePO4.traj")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = CuratorTorchSimAdapter(
        "../best_model.ckpt",
        compute_neighbor_list=True,
        cutoff=None,  # infer from model if available
        detach=True,
        device=device,
        load_compiled=False,
    )

    def make_energy_logger():
        counter = {"step": -1}

        def _log_energy(state, model):
            counter["step"] += 1
            energy = getattr(state, "energy", None)
            if energy is None and model is not None:
                energy = model(state)["energy"]
            val = energy.detach().cpu().view(-1).tolist()
            print(f"[step {counter['step']}] energy={val}")
            return energy

        return _log_energy

    reporter = TrajectoryReporter(
        filenames="md_report.h5",
        state_frequency=1,
        prop_calculators={1: {"energy": make_energy_logger()}},
        metadata={"note": "curator+torch-sim demo"},
    )

    start = time.time()

    final_state = integrate(
        atoms,
        adapter,
        integrator=nve,
        n_steps=1000,
        temperature=300.0,
        timestep=1e-3,
        trajectory_reporter=reporter,
        pbar=False,
    )

    print("Final potential energy (eV):", final_state.energy)
    print("Final forces shape:", final_state.forces.shape)
    print("Trajectory written to: md_report.h5")
    print(f"Time for this simulation: {time.time() - start} s")

if __name__ == "__main__":
    main()
