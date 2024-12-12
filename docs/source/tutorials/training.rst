.. _ASE: https://wiki.fysik.dtu.dk/ase/install.html
.. _VASP: https://www.vasp.at/
.. _PaiNN: https://arxiv.org/abs/2102.03150
.. _NequIP: https://www.nature.com/articles/s41467-022-29939-5
.. _MACE: https://arxiv.org/abs/2206.07697

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
And a ``config.yaml`` file containing all hyperparameters used in the task will also be generated once the user run any jobs. The users can refer to this file for setting up their own training tasks.
Except for training, all other tasks can be initiated in both ways.

Using hydra's defaults list
----------------------------
Now there are three equivariant graph neural network architechtures (PaiNN_, NequIP_, MACE_) implemented in ``curator``. 
Changing one from another may need one to modify a lot of model hyperparameters, and this could be a painful task for people who are not familiar with the models and ``curator`` codebase.

Using hydra's defaults list can make this process much easier. You can specify the default configs that defines the default parameters for different architechtures by changing ``model/representation``:

.. code-block:: yaml
    
    defaults:
      - model/representation: mace

Likewise, you can specify other default config files in ``curator/configs``. For example, we can change the default logger for trainer, use different loss functions, specify different types of error metrics by:

.. code-block:: yaml
    
    defaults:
      - task/outputs: energy_force_virial
      - trainer/logger: wandb

Initiating objects
------------------
Sometimes you may want to create the model or dataset in python with the config file. It can be done by using hydra's instantiate function.

.. code-block:: python
    :linenos:

    from hydra.utils import instantiate
    from curator.utils import read_user_config

    cfg = read_user_config('user_cfg.yaml', config_name='train.yaml')  # for different tasks using different config_name. For example, select.yaml for selection.
    model = instantiate(cfg.model)
    data = instantiate(cfg.data)
    data.setup()

Hydra supports recursive instantiation and you can override parameters during instantiation:

.. code-block:: python
    :linenos:

    data = instantiate(cfg.data, batch_size=32)

More usage examples can be found in https://hydra.cc/docs/advanced/instantiate_objects/overview/

Preprocessing dataset
---------------------
In **CURATOR**, preprocessing dataset includes but not restricts to: 1. Dataset normalization. 2. Unit transform. 3 Compute neighbor list. 4. Cast data type.

In most cases, preprocessing dataset may significantly improve the model performance. Especially for the cases where the atomic energies are far away from 0 and the dataset containing multiple types of structures.

We provided several normalization schemes for energies.

1. ``data.atomwise_``Atomwise normalization which calculates the atomic . 
2. Structure-based Normalization. This option is often not suggested because it may cause worse predictions for structures with different number of atoms.
3. Per-species normalization.
4. Reference energies.
5. Scale forces

Multi-GPU training
------------------