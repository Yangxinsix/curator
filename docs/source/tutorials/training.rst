.. _ASE: https://wiki.fysik.dtu.dk/ase/install.html
.. _VASP: https://www.vasp.at/
.. _PaiNN: https://arxiv.org/abs/2102.03150
.. _NequIP: https://www.nature.com/articles/s41467-022-29939-5
.. _MACE: https://arxiv.org/abs/2206.07697
.. _Hydra: https://hydra.cc/
.. _PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/

Training Machine Learning Potentials
====================================

This tutorial explains how to train machine learning potentials using CURATOR. CURATOR uses `Hydra`_ for managing hyperparameters and configuration files, allowing you to leverage a wide range of Hydra features for flexible experimentation.

Training and evaluation are powered by the `PyTorch Lightning`_ framework, enabling the use of advanced deep learning features with minimal effort.

Preparing Data
.. _Hydra: https://hydra.cc/
.. _PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/

Training Machine Learning Potentials
====================================

This tutorial explains how to train machine learning potentials using CURATOR. CURATOR uses `Hydra`_ for managing hyperparameters and configuration files, allowing you to leverage a wide range of Hydra features for flexible experimentation.

Training and evaluation are powered by the `PyTorch Lightning`_ framework, enabling the use of advanced deep learning features with minimal effort.

Preparing Data
--------------

Training equivariant neural network potentials is beginner-friendly with CURATOR. All hyperparameters are pre-defined except for the ``datapath``. Therefore, the only required input to train a model is the ab-initio dataset.

CURATOR uses ASE's API to read atomistic structures, so any format supported by ASE_ is compatible. For example, you can use an **OUTCAR** file from a VASP_ calculation or an ASE-compatible ``.traj`` file.

A minimal training task can be launched with:

Training equivariant neural network potentials is beginner-friendly with CURATOR. All hyperparameters are pre-defined except for the ``datapath``. Therefore, the only required input to train a model is the ab-initio dataset.

CURATOR uses ASE's API to read atomistic structures, so any format supported by ASE_ is compatible. For example, you can use an **OUTCAR** file from a VASP_ calculation or an ASE-compatible ``.traj`` file.

A minimal training task can be launched with:

.. code-block:: bash

    curator-train data.datapath=water.traj

This employs CURATOR's default settings, using the PaiNN_ model architecture. If you want to customize hyperparameters, you can either:

- Use the command-line override:

  .. code-block:: bash

      curator-train model/representation=nequip

  CURATOR's NequIP representation now follows the NequIP 0.17 model layout while
  keeping CURATOR's config names. In practice this means:

  - ``num_interactions`` maps to NequIP's ``num_layers``
  - ``lmax`` maps to ``l_max``
  - ``num_basis`` maps to ``num_bessels``
  - ``power`` maps to ``polynomial_cutoff_p``
  - ``num_features`` can be either a single integer or a list of length ``lmax + 1``

  The current CURATOR NequIP defaults are closer to the upstream NequIP 0.17 baseline:
  ``num_interactions=4``, ``lmax=1``, ``radial_mlp_width=128``. Optional
  ``type_embed_num_features`` and readout MLP settings are also exposed directly in
  ``model.representation``.

- Or specify them in a separate YAML configuration file:
This employs CURATOR's default settings, using the PaiNN_ model architecture. If you want to customize hyperparameters, you can either:

- Use the command-line override:

  .. code-block:: bash

      curator-train model/representation=nequip

- Or specify them in a separate YAML configuration file:

  .. code-block:: bash
  .. code-block:: bash

      curator-train cfg=user_cfg.yaml
      curator-train cfg=user_cfg.yaml

You can define any parameters in the config file. For reference, the meanings of different parameters are documented under ``curator/configs/``. Once a job is run, a complete ``config.yaml`` file will be generated. This file contains the effective configuration and can be reused or modified for future tasks.

All other CURATOR tasks (e.g., selection, evaluation) can be initiated in the same way.
You can define any parameters in the config file. For reference, the meanings of different parameters are documented under ``curator/configs/``. Once a job is run, a complete ``config.yaml`` file will be generated. This file contains the effective configuration and can be reused or modified for future tasks.

All other CURATOR tasks (e.g., selection, evaluation) can be initiated in the same way.

Using Hydra's Defaults List
---------------------------

CURATOR supports multiple equivariant GNN architectures, including PaiNN_, NequIP_, and MACE_. Switching between them usually requires updating many hyperparameters, which can be cumbersome.

Hydra's ``defaults`` list allows you to easily switch architectures by referencing a pre-defined configuration:
Using Hydra's Defaults List
---------------------------

CURATOR supports multiple equivariant GNN architectures, including PaiNN_, NequIP_, and MACE_. Switching between them usually requires updating many hyperparameters, which can be cumbersome.

Hydra's ``defaults`` list allows you to easily switch architectures by referencing a pre-defined configuration:

.. code-block:: yaml


    defaults:
      - model/representation: mace

Similarly, you can configure other components such as loss functions, loggers, and evaluation metrics. For example:
Similarly, you can configure other components such as loss functions, loggers, and evaluation metrics. For example:

.. code-block:: yaml


    defaults:
      - task/outputs: energy_force_virial
      - trainer/logger: wandb

Instantiating Objects in Python
-------------------------------

You may wish to create models or datasets programmatically from a config file. CURATOR supports this via Hydra's ``instantiate`` function.
Instantiating Objects in Python
-------------------------------

You may wish to create models or datasets programmatically from a config file. CURATOR supports this via Hydra's ``instantiate`` function.

.. code-block:: python
    :linenos:

    from hydra.utils import instantiate
    from curator.utils import read_user_config

    cfg = read_user_config("user_cfg.yaml", config_name="train.yaml")
    cfg = read_user_config("user_cfg.yaml", config_name="train.yaml")
    model = instantiate(cfg.model)
    data = instantiate(cfg.data)
    data.setup()

Hydra supports recursive instantiation and parameter overrides:
Hydra supports recursive instantiation and parameter overrides:

.. code-block:: python
    :linenos:

    data = instantiate(cfg.data, batch_size=32)

More examples can be found in the `Hydra instantiate documentation <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_.

Saving model and restart training
--------------------------------- 
CURATOR automatically saves model checkpoints during training using PyTorch Lightning's built-in ``ModelCheckpoint`` callback. By default, it saves the latest and best-performing checkpoints according to validation loss.

The checkpoints will be saved to ``model_path/best_model_{epoch}_{step}_{val_total_loss:.2f}.ckpt``.

You can also customize checkpointing behavior in the configuration file, for example to save checkpoints for every ``N`` epoch:

.. code-block:: yaml

    trainer:
      callbacks:
        - _target_: pytorch_lightning.callbacks.ModelCheckpoint
          dirpath: model_path
          save_top_k: -1
          every_n_epochs: 10  # can be any integers

To resume training from a specific checkpoint:

.. code-block:: bash

    curator-train model_path=model_path/best_model_epoch=10.ckpt

.. Preprocessing Datasets
.. ----------------------

.. Dataset preprocessing in CURATOR includes (but is not limited to):

.. 1. Dataset normalization  
.. 2. Unit conversion  
.. 3. Neighbor list construction  
.. 4. Data type casting

.. Proper preprocessing often significantly improves model performance, especially when:

.. - Atomic energies are far from zero
.. - The dataset contains diverse structures

.. CURATOR supports several normalization schemes for energies:

.. 1. **Atomwise normalization**: Normalizes energy per atom.  
.. 2. **Structure-based normalization**: Not recommended, as it can degrade performance on systems with varying atom counts.  
.. 3. **Per-species normalization**: Adjusts normalization by chemical species.  
.. 4. **Reference energy subtraction**: Subtracts fixed reference values.
.. 5. **Force scaling**: Adjusts force magnitude for training stability.

Multi-GPU Training
Multi-GPU Training
------------------

PyTorch Lightning makes multi-GPU training easy and scalable. To enable multi-GPU training, modify the following options in your YAML file:


1. Specify Number of GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~

Set the number of GPUs using the ``trainer.devices`` option.

2. Choose the Distributed Strategy (Recommended: ``ddp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the Distributed Data Parallel (DDP) strategy by setting ``trainer.strategy`` to ``"ddp"``. This is the most commonly used and recommended strategy for multi-GPU training.

3. Adjust Batch Size
~~~~~~~~~~~~~~~~~~~~~

Multiply your original batch size by the number of GPUs so that each GPU processes the same amount of data as in single-GPU training.

4. Enable Mixed Precision (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For faster training and lower memory usage, enable automatic mixed precision by setting ``trainer.precision`` to ``"16-mixed"`` (for PyTorch >= 1.6).

.. code-block:: yaml

    trainer:
      devices: <number_of_gpus>
      strategy: ddp
      precision: 16-mixed  # Optional: enables mixed precision

    data:
      batch_size: <batch_size_per_gpu> * <number_of_gpus>

.. note::

    For example, if you previously used ``batch_size: 32`` on a single GPU, and now want to use 4 GPUs, set ``batch_size: 128``.

``ddp`` ensures each GPU handles a portion of the data in parallel and synchronizes gradients efficiently. Mixed precision further speeds up training while reducing GPU memory usage.

cuEquivariance acceleration
---------------------------

cuEquivariance is an NVIDIA Python library to accelerate GPU kernel for graph neural networks. It can speed up training and inference of MACE and nequip models significantly.
To enable cuEquivariance acceleration in CURATOR, simply set:

.. code-block:: yaml

    model:
      representation:
        use_cueq: true

This will automatically apply cuEquivariance optimized layers where available.
