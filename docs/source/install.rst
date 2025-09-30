Installation
============

This guide explains how to install **CURATOR** and its optional acceleration tools.

Requirements
------------

- `Python <https://www.python.org/>`_ >= 3.8  
- `PyTorch <https://pytorch.org/get-started/locally/>`_ >= 1.9  

Install via pip
---------------

.. code-block:: bash

    pip install --upgrade pip
    pip install curator_torch

This installs the **basic version** of CURATOR.  
For better training and inference performance, install additional acceleration packages:

.. code-block:: bash

    # Install asap3, matscipy, and torch_scatter
    pip install curator_torch[opt]

    # Install CuEquivariance
    pip install curator_torch[cueq]

    # Install all optional acceleration tools
    pip install curator_torch[all]

Install from source
-------------------

To install the latest version:

.. code-block:: bash

    git clone https://github.com/Yangxinsix/curator.git
    cd curator
    pip install .

To include all acceleration tools:

.. code-block:: bash

    pip install .[all]

Installing PyTorch with CUDA
----------------------------

If PyTorch is installed on a login node without GPU support, the **CPU-only** version may be installed by default.  
To explicitly install a CUDA-enabled version of PyTorch, specify the CUDA toolkit version:

.. code-block:: bash

    conda install pytorch=*=*cuda* pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia

Now you are ready to use **CURATOR**!
