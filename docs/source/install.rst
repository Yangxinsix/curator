Installation
=============

Requirements
-------------
- `Python <https://www.python.org/>`_ (>=3.8)
- `Pytorch <https://pytorch.org/get-started/locally/>`_ (>=1.9)
- `ASE <https://wiki.fysik.dtu.dk/ase/install.html>`_ (>=3.22.0)
- `Myqueue <https://myqueue.readthedocs.io/en/latest/installation.html>`_ (==22.7.1)
- `e3nn <https://e3nn.org/>`_
- `Hydra <https://hydra.cc/>`_
- `PyTorch Lightning <https://github.com/Lightning-AI/pytorch-lightning>`_

pip installation
---------------------
.. code-block:: bash
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    git clone https://github.com/Yangxinsix/curator.git
    pip install ./curator


conda installation
----------------------
After installed all required packages, installing the code can be quite simple.

.. code-block:: bash
    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    git clone https://github.com/Yangxinsix/curator.git
    pip install ./curator


Sometimes you may run the code on the login node which may not be equiped with GPU. If you are using above commands to install PyTorch, cpu-only version will be installed.
To install the cuda version pytorch, you need to specify the cuda version run the following command:

.. code-block:: bash
    conda install pytorch=*=*cuda* cudatoolkit==11.8 -c pytorch


Now you are ready to go!