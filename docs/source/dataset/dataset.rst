Prepare the dataset
==============================

To perform efficient *ab initio* electronic structure calculation by DeepH method 
for a class of large-scale material systems, one needs to design an appropriate 
dataset of small structures that have close chemical bonding environment with 
the target large-scale material systems. Therefore, the first step of a DeepH 
study is to perform the DFT calculation on the above dataset to get the DFT 
Hamiltonian matrices with the localized basis. DeepH-pack supports DFT
results made by ABACUS, OpenMX, FHI-aims or SIESTA and will support HONPAS
soon.

Using ABACUS
^^^^^^^^^^^^^^^^^^^^^^^^

One needs to perform the DFT calculation with ABACUS
to get the Kohn-Sham Hamiltonian output file in the csr
format. This output file should be placed in a separate
folder for each structure in the dataset. In order to get
this csr file, the input file of ABACUS should include
keywords like this::

    out_mat_hs2   1

Note added: the DeepH-ABACUS interface currently suffers from bug regarding the sparsity pattern of ABACUS's overlap matrix, which may cause errors in DeepH prediction. We're currently working on this issue, and this note will be removed once a fix is ready.

Using OpenMX
^^^^^^^^^^^^^^^^^^^^^^^^

One needs to perform the DFT calculation with OpenMX 
to get the Kohn-Sham Hamiltonian output file in a binary 
form. This binary file should be placed in a separate 
folder for each structure in the dataset and should be 
named as ``openmx.scfout``. In order to get this binary file, 
the input file of OpenMX should include keywords like this::

    System.Name   openmx
    HS.fileout    On

Besides, it is required to attach the text output of 
``openmx.out`` to the end of ``openmx.scfout``, which 
means to run::

    cat openmx.out >> openmx.scfout

Using FHI-aims
^^^^^^^^^^^^^^^^^^^^^^^^

One needs to perform the DFT calculation with modified FHI-aims
to get the Kohn-Sham Hamiltonian output file in text
format. This output file should be placed in a separate
folder for each structure in the dataset.

Using SIESTA
^^^^^^^^^^^^^^^^^^^^^^^^

One needs to perform DFT calculation with SIESTA to get Hamiltonians in binary
file named ``${System_name}.HSX``. To activate this feature, you should include 
keyword in ``${System_name}.fdf`` file::

    SaveHS true

It is also recommended to specify a higher convergence criteria for SCF calculation.
We found it sufficient to write in ``${System_name}.fdf`` file::

    DM.Tolerence  1.d-9
