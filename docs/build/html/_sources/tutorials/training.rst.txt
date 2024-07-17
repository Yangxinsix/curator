.. _ASE: https://wiki.fysik.dtu.dk/ase/install.html
.. _VASP: https://www.vasp.at/
.. _PaiNN: https://arxiv.org/abs/2102.03150

Training machine learning potentials
=================================================
This tutorial will explain how to train models using CURATOR.

Preparing data
--------------
Training equivariant neural network potentials can be super easy with `curator` for new beginners. We defined every default hyperparameters except the `datapath`.
Therefore, the only thing you need to train a new model is the ab-initio dataset. CURATOR uses ASE's api for reading atomistic data, so any formats supported by ASE will work in `curator`.

For example, you can provide an **OUTCAR** file from VASP_ calculation or use ASE_ recommended `.traj` file as the training dataset.
A simple training task can be executed by running:

.. code-block:: bash

    curator-train data.datapath=water.traj

The parameters will use the default settings in CURATOR and the default GNN architechture is PaiNN_ model. 
If you want to specify hyperparameters by yourself, you can choose to use command line interface like ``curator-train model/representation=nequip``, or resort to external configuration file like:

.. code-block:: bash

    curator-train cfg=user_cfg.yaml

You can select any parameters you want to use inside this file. The meanings of different parameters can be found in ``curator/configs`` folder. 
And a `config.yaml` file containing all hyperparameters used in the task will also be generated once the user run any jobs. The users can refer to this file for setting up their own training tasks.
Except for training, all other tasks can be initiated in both ways.