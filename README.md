# <div align="center">CURATOR: Building Robust Machine Learning Potentials for Atomistic Simulations</div>

**CURATOR** is an autonomous active learning workflow for constructing equivariant machine-learned interatomic potentials (MLIPs).  
It currently supports three message-passing neural network (MPNN) architectures:  
- [PAINN](https://arxiv.org/abs/2102.03150)  
- [NequIP](https://arxiv.org/abs/2101.03164)  
- [MACE](https://arxiv.org/abs/2206.07697)

---

## Features
- **Equivariant graph neural networks**: built-in support for PAINN, NequIP, and MACE.  
- **Active learning loop**: train → simulate → select → label, with batch active learning sampling.  
- **Workflow automation**: seamless integration with [myqueue](https://myqueue.readthedocs.io) for job scheduling and workflow management.  
- **High-performance computing support**: multi-GPU training and inference, GPU kernel acceleration with [CuEquivariance](https://github.com/NVIDIA/cuEquivariance).  
- **Integration with MD softwares**: ASE calculator integration and LAMMPS interface that enable versatile simulations. 
- **Extensible design**: easy to add new MLIP models, adapt to custom datasets or implement versatile simulations.

---

## Installation

### Requirements
- Python >= 3.8  
- [PyTorch >= 1.13](https://pytorch.org/get-started/locally/)  

> ⚠️ Please follow the official PyTorch installation guide to select the correct version for your operating system, CUDA, or CPU setup.

### From PyPI
```bash
pip install --upgrade pip
pip install curator_torch
```

### From source
```bash
git clone https://github.com/Yangxinsix/curator.git
cd curator 
pip install .
```

## Documentation
A documentation is available at: https://curator-gnn.readthedocs.io/en/latest/

## Quick Start
CURATOR organizes the ML potential construction workflow into four modular steps:
1. Train – train an MLIP on quantum computational reference data.
2. Simulate – perform molecular dynamics (MD), nudged elastic band (NEB) simulations or many other simulations.
3. Select – identify new informative configurations based on batch active learning algorithms.
4. Label – compute reference labels (e.g., with VASP) for selected configurations
Each step can be run independently. For autonomous execution, CURATOR integrates with [myqueue](https://myqueue.readthedocs.io) and a user-defined configuration file to manage and chain all procedures together.
CURATOR contains four procedures for constructing a machine learning potential, including train, simulation, select and label. Please refer to the [curator documentation](https://curator-gnn.readthedocs.io/en/latest/) for more details.

A minimal working example is provided in the `/example` directory. This demonstrates the molecular dynamics of LiFePO₄.


A working example is presented in `/example` where you will model the diffusivity of LiFePO4 using both MD simulation and NEB. 
First you download the curator package as described above. Then you create a directory somewhere. You then need to copy the user configuration script `user_cfg.yaml` , the inital dataset `init_dataset.traj`, the MD simulation trajectories `LiFePO4_MD_0.traj`,`LiFePO4_MD_1.traj`,`LiFePO4_MD_4.traj`, and the initial and final images for the NEB `NEB_init_pristine.traj` and `NEB_final_pristine.traj`(You can also optimize these NEB structures yourself if you want you). You need to change the datapaths in the user configuration file such that it matches your directory. To run the workflow you need to have a myqueue configuration folder and file `/.myqueue/config.py`. It can also be downloaded from the example case, but it should be customized to your HPC or local computer. To run the workflow on your HPC please change the nodename and cores in `user_cfg.yaml` for each task. To run the workflow you either need to copy the workflow script `curator-workflow` from the exmaple folder into the same diretcory as `user_cfg.yaml` or locate the path to the script in `Curator/scripts`. You then write `mq workflow curator-workflow` in the terminal and the workflow will starts. A more illustrative example and video tutorial will be published soon. 

There are a couple of thing to note. First if you want to run [VASP](https://www.vasp.at/) in the labeling script you need to load a license version or else we recommend you to use [GPAW](https://wiki.fysik.dtu.dk/gpaw/). Secondly, in the end of each iteration you need to add the data to the initial dataset your self. Thirdly, if you do not want to train your model from scratch in the next iteration you should use the load_model paramater in `user_cfg.yaml` to load the previous iteration's model

If you want to dig into the code you can find all the working functions in `Curator/curator/cli.py` and to understand how the data was generated for the example case you can go to `Curator/example/Datageneration`

## <div align="left">Reference</div>
If you use CURATOR in your research, please cite:
https://chemrxiv.org/engage/chemrxiv/article-details/65cd6a5366c1381729ab0854