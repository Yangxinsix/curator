# <div align="center">CURATOR: Building Robust Machine Learning Potentials for Atomistic Simulations</div>
This package implements an autonomous active learning workflow for construction of equivarient Machine-learned interatomic potentials (MLIPs). In this workflow you can choose between three architechtures of message passing neural networks (MPNN): [PAINN](https://arxiv.org/abs/2102.03150), [NequiP](https://arxiv.org/abs/2101.03164) or [MACE](https://arxiv.org/abs/2206.07697).

To acquire a more accurate MLIP batch active learning is used. By first simulating a particular structure, batch active learning picks out the most diverse and uncertain structures to be labelled. The learned features or gradients in the model are used for active learning. Several selection methods are implemented.  
All the active learning codes are to be tested.
The labelled structures are added to the dataset to train a more accurate MLIP.

Before training your MLIP you need to acquire an initial dataset consisting of atomic structures. That can be a collection of molecular dynamic (MD) simulation trajectories, ionic steps from an atomistic optimization, nudged elastic band (NEB) and much more. The important thing is to keep the level of theory consistent for all data meaning you need to use the same density funtional theory (DFT) calcululator or at least one with similar level of theory. When you have acquired your initial data set, you need to combine all your data to one big trajectory file saved using [ASE](https://iopscience.iop.org/article/10.1088/1361-648X/aa680e) trajectory format (example: "database.traj").

## <div align="left">Documentation</div>
The code itself is well documented and a working example is presented. A more indepth documentation has yet to be made. You can find all hyperparameters for the workflow and how they work in our default configuration folder.

## <div align="left">Quick Start</div>


## <div align="left">How to install</div>

This code is only tested on [**Python>=3.8.0**](https://www.python.org/) and [**PyTorch>=2.0**](https://pytorch.org/get-started/locally/).  
Requirements: [PyTorch Lightning](https://lightning.ai/), [ASE](https://wiki.fysik.dtu.dk/ase/index.html),
[hydra](https://hydra.cc/), [myqueue](https://myqueue.readthedocs.io/en/latest/installation.html)(if you want to submit jobs automatically).

```
$ conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
$ git clone https://gitlab.gbar.dtu.dk/swano/Curator.git
$ cd Curator 
$ pip install .
```

## <div align="left">How to use</div>
A working example is presented in `/example` where you will model the diffusivity of LiFePO4 using both MD simulation and NEB. 
First you download the curator package as described above. Then you create a directory somewhere. You then need to copy the user configuration script `user_cfg.yaml` , the inital dataset `init_dataset.traj`, the MD simulation trajectories `LiFePO4_MD_0.traj`,`LiFePO4_MD_1.traj`,`LiFePO4_MD_4.traj`, and the initial and final images for the NEB `NEB_init_pristine.traj` and `NEB_final_pristine.traj`(You can also optimize these NEB structures yourself if you want you). You need to change the datapaths in the user configuration file such that it matches your current directory. To run the workflow you need to have a myqueue configuration folder and file `/.myqueue/config.py`. It can also be downloaded from the example case, but it should be customized to your HPC or local computer. To run the workflow on your HPC please change the nodename and cores in `user_cfg.yaml` for each task. To run the workflow you either need to copy the workflow script `curator-workflow` from the exmaple folder into the same diretcory as `user_cfg.yaml` or locate the path to the script in `Curator/scripts`. You then write `mq workflow curator-workflow` in the terminal and the workflow will starts. A more illustrative example and video tutorial will be published soon. 

There are a couple of thing to note. First if you want to run [VASP](https://www.vasp.at/) in the labeling script you need to load a license version or else we recommend you to use [GPAW](https://wiki.fysik.dtu.dk/gpaw/). Secondly, in the end of each iteration you need to add the data to the initial dataset your self. Thirdly, if you do not want to train your model from scratch in the next iteration you should use the load_model paramater in `user_cfg.yaml` to load the previous iteration's model

If you want to dig into the code you can find all the working functions in `Curator/curator/cli.py` and to understand how the data was generated for the example case you can go to `Curator/example/Datageneration`

## <div align="left">Reference</div>
If you are using the code for building MLIPs, please cite:
https://chemrxiv.org/engage/chemrxiv/article-details/65cd6a5366c1381729ab0854