Installation
============

Requirements
------------

To use DeepH-pack, following environments and packages are required:

Python packages
^^^^^^^^^^^^^^^^^^^^^^^^

Prepare the Python 3.9 interpreter. Install the following Python packages required:

* NumPy
* SciPy
* PyTorch = 1.9.1
* PyTorch Geometric = 1.7.2
* e3nn = 0.3.5
* pymatgen
* h5py
* TensorBoard
* pathos
* psutil

In Linux, you can quickly achieve the requirements by running::

   # install miniconda with python 3.9
   wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
   bash Miniconda3-py39_4.10.3-Linux-x86_64.sh

   # install packages by conda
   conda install numpy
   conda install scipy
   conda install pytorch==1.9.1 ${pytorch_config}
   conda install pytorch-geometric=1.7.2 -c rusty1s -c conda-forge
   conda install pymatgen -c conda-forge

   # install packages by pip
   pip install e3nn==0.3.5
   pip install h5py
   pip install tensorboard
   pip install pathos
   pip install psutil

with ``${pytorch_config}`` replaced by your own configuration. 
You can find how to set it in `the official website of PyTorch <https://pytorch.org/get-started/previous-versions/>`_.

Julia packages
^^^^^^^^^^^^^^^^^^^^^^^^

Prepare the Julia 1.5.4 interpreter. Install the following Julia packages required with Julia's builtin package manager:

* Arpack.jl
* HDF5.jl
* ArgParse.jl
* JLD.jl
* JSON.jl
* IterativeSolvers.jl
* DelimitedFiles.jl
* StaticArrays.jl
* LinearMaps.jl
* Pardiso.jl

In Linux, you can quickly achieve the requirements by first running::

   # install julia 1.6.6
   wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.6-linux-x86_64.tar.gz
   tar xzvf julia-1.6.6-linux-x86_64.tar.gz

   # open the julia REPL
   julia

Then enter the pkg REPL by pressing ``]`` from the Julia REPL. In the pkg REPL run::

   (@v1.6) pkg> add Arpack
   (@v1.6) pkg> add HDF5
   (@v1.6) pkg> add ArgParse
   (@v1.6) pkg> add JLD
   (@v1.6) pkg> add JSON
   (@v1.6) pkg> add IterativeSolvers
   (@v1.6) pkg> add DelimitedFiles
   (@v1.6) pkg> add StaticArrays
   (@v1.6) pkg> add LinearMaps

Follow `these instructions <https://github.com/JuliaSparse/Pardiso.jl#mkl-pardiso>`_ to install Pardiso.jl.

Install DeepH-pack
------------------------

Run the following command in the path of DeepH-pack::
   
   git clone https://github.com/mzjb/DeepH-pack.git
   cd DeepH-pack
   pip install .


Install one of the supported DFT packages
------------------------------------------------

One of the supported DFT packages is required to obtain the dataset.
DeepH-pack supports DFT results made by ABACUS, OpenMX, FHI-aims or SIESTA,
and will support HONPAS soon.

ABACUS
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   abacus

OpenMX
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   
   openmx

FHI-aims
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   aims

SIESTA
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   siesta

