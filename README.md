<p align="center">
  <img height="90" src="logo/logo_word.svg"/>
</p>

--------------------------------------------------------------------------------
[![DOI:10.1038/s43588-022-00265-6](https://zenodo.org/badge/DOI/10.1038/s43588-022-00265-6.svg)](https://doi.org/10.1038/s43588-022-00265-6)
[![Documentation Status](https://readthedocs.org/projects/deeph-pack/badge/)](https://deeph-pack.readthedocs.io/)

DeepH-pack is the official implementation of the Deep Hamiltonian (DeepH) method described in the paper [*Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation*](https://www.nature.com/articles/s43588-022-00265-6) and in the [Research Briefing](https://www.nature.com/articles/s43588-022-00270-9).

DeepH-pack supports DFT results made by
[ABACUS](https://abacus.ustc.edu.cn/), [OpenMX](http://www.openmx-square.org/)
or [FHI-aims](https://fhi-aims.org/)
and will support [SIESTA](https://departments.icmab.es/leem/siesta/) and
[HONPAS](http://honpas.ustc.edu.cn/) soon.

For more information, see the
[documentation](https://deeph-pack.readthedocs.io).

# Contents
1. [How to cite](#how-to-cite)
1. [Requirements](#requirements)
1. [Usage](#usage)
1. [Demo](#demo-deeph-study-on-twisted-bilayer-bismuthene)


## How to cite

```
@article{deeph,
   author = {Li, He and Wang, Zun and Zou, Nianlong and Ye, Meng and Xu, Runzhang and Gong, Xiaoxun and Duan, Wenhui and Xu, Yong},
   title = {Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation},
   journal = {Nature Computational Science},
   volume = {2},
   number = {6},
   pages = {367-377},
   ISSN = {2662-8457},
   DOI = {10.1038/s43588-022-00265-6},
   url = {https://doi.org/10.1038/s43588-022-00265-6},
   year = {2022},
   type = {Journal Article}
}
```

## Requirements

To use DeepH-pack, following environments and packages are required：

### Python
Prepare the Python 3.9 interpreter. Install the following Python packages required:
- NumPy
- SciPy
- PyTorch = 1.9.1
- PyTorch Geometric = 1.7.2
- e3nn = 0.3.5
- pymatgen
- h5py
- TensorBoard
- pathos
- psutil

In Linux, you can quickly achieve the requirements by running

```bash
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
```

with `${pytorch_config}` replaced by your own configuration. 
You can find how to set it in [the official website of PyTorch](https://pytorch.org/get-started/previous-versions/).
### Julia
Prepare the Julia 1.5.4 interpreter. Install the following Julia packages required with Julia's builtin package manager:
- Arpack.jl
- HDF5.jl
- ArgParse.jl
- JLD.jl
- JSON.jl
- IterativeSolvers.jl
- DelimitedFiles.jl
- StaticArrays.jl

In Linux, you can quickly achieve the requirements by first running
```bash
# install julia 1.5.4
wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.4-linux-x86_64.tar.gz
tar xzvf julia-1.5.4-linux-x86_64.tar.gz

# open the julia REPL
julia
```
Then enter the pkg REPL by pressing `]` from the Julia REPL. In the pkg REPL run
```julia
(@v1.5) pkg> add Arpack
(@v1.5) pkg> add HDF5
(@v1.5) pkg> add ArgParse
(@v1.5) pkg> add JLD
(@v1.5) pkg> add JSON
(@v1.5) pkg> add IterativeSolvers
(@v1.5) pkg> add DelimitedFiles
(@v1.5) pkg> add StaticArrays
```

### One of the supported DFT packages
One of the supported DFT packages is required to obtain the dataset and
calculate the overlap matrix for large-scale material systems.
DeepH-pack supports DFT results made by ABACUS, OpenMX or FHI-aims
and will support SIESTA soon.

1. **ABACUS**: Install [ABACUS package](https://abacus.ustc.edu.cn)
    for density functional theory Hamiltonian matrix calculation to
    construct datasets. DeepH-pack requires
    [ABACUS version >= 2.3.2](https://github.com/deepmodeling/abacus-develop/releases/tag/v2.3.2).
2. **OpenMX**:
    1. Install [OpenMX package version 3.9](http://www.openmx-square.org/download.html) for density functional theory Hamiltonian matrix calculation to construct datasets.
        If you are using Intel MKL and Intel MPI environments, you can use the following variable definitions for makefile
        ```
        CC = mpiicc -O3 -xHOST -ip -no-prec-div -qopenmp -I${MKLROOT}/include/fftw -I${MKLROOT}/include
        FC = mpiifort -O3 -xHOST -ip -no-prec-div -qopenmp -I${MKLROOT}/include
        LIB = ${CMPLR_ROOT}/linux/compiler/lib/intel64_lin/libiomp5.a ${MKLROOT}/lib/intel64/libmkl_blas95_lp64.a ${MKLROOT}/lib/intel64/libmkl_lapack95_lp64.a ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group ${CMPLR_ROOT}/linux/compiler/lib/intel64_lin/libifcoremt.a -lpthread -lm -ldl
        ```
        Or edit the makefile yourself according to your environment to install OpenMX version 3.9.
    2. A modified OpenMX package is also used to compute overlap matrices only for large-scale materials structure. Install 'overlap only' OpenMX according to the *readme* documentation in this [repository](https://github.com/mzjb/overlap-only-OpenMX).
## Usage

### Install DeepH-pack
Run the following command in the path of DeepH-pack:
```bash
git clone https://github.com/mzjb/DeepH-pack.git
cd DeepH-pack
pip install .
```

### Prepare the dataset
To perform efficient *ab initio* electronic structure calculation by DeepH method 
for a class of large-scale material systems, one needs to design an appropriate 
dataset of small structures that have close chemical bonding environment with 
the target large-scale material systems. Therefore, the first step of a DeepH 
study is to perform the DFT calculation on the above dataset to get the DFT 
Hamiltonian matrices with the localized basis. DeepH-pack supports DFT 
results made by ABACUS, OpenMX or FHI-aims and will support SIESTA and HONPAS soon.

For more information, see the
[documentation](https://deeph-pack.readthedocs.io/en/latest/dataset/dataset.html).

### Preprocess the dataset
`Preprocess` is a part of DeepH-pack. Through `Preprocess`, DeepH-pack will
convert the unit of physical quantity, store the data files in the format
of text and *HDF5* for each structure in a separate folder, generate local
coordinates, and perform basis transformation for DFT Hamiltonian matrices.
We use the following convention of units:

Quantity | Unit 
---|---
Length   | Å    
Energy   | eV   

You need to edit a configuration in the format of *ini*, setting up the
file referring to the default file `DeepH-pack/deeph/preprocess/preprocess_default.ini`.
The meaning of the keywords can be found in the
[documentation](https://deeph-pack.readthedocs.io/en/latest/keyword/preprocess.html).
For a quick start, you must set up *raw_dir*, *processed_dir* and *interface*.

With the configuration file prepared, run 
```bash
deeph-preprocess --config ${config_path}
```
with `${config_path}` replaced by the path of your configuration file.

### Train your model
`Train` is a part of DeepH-pack, which is used to train a deep learning model using the processed dataset.

Prepare a configuration in the format of *ini*, setting up the file referring to the default `DeepH-pack/deeph/default.ini`. The meaning of the keywords can be found in the [documentation](https://deeph-pack.readthedocs.io/en/latest/keyword/train.html). For a quick start, you must set up *graph_dir*, *save_dir*, *raw_dir* and *orbital*, other keywords can stay default and be adjusted later.

With the configuration file prepared, run 
```bash
deeph-train --config ${config_path}
```
with `${config_path}` replaced by the path of your configuration file.

Tips:
- **Name your dataset**. Use *dataset_name* to name your dataset, the same names may overwrite each other.

- **Hyperparameters of the neural network**. The neural network here contains some hyperparameters. For a specific problem your should try adjusting the hyperparameters to obtain better results.

- **The keyword *orbital***. The keyword *orbital* states which orbitals or matrix elements are predicted. It is a little complicated to understand its data structure. To figure out it, you can refer to the [documentation](https://deeph-pack.readthedocs.io/en/latest/keyword/train.html#:~:text=generate%20crystal%20graphs.-,orbital,-%3A%20A%20JSON%20format) or the method [make_mask](https://github.com/mzjb/DeepH-pack/blob/main/deeph/kernel.py#:~:text=def%20make_mask(self%2C%20dataset)%3A) in class `DeepHKernal` defined in `DeepH-pack/deeph/kernel.py`.

    Alternatively, a Python script at `DeepH-pack/tools/get_all_orbital_str.py` can be used to generate a default configuration to predict all orbitals with one model.

- **Use TensorBoard for visualizations**. You can track and visualize the training process through TensorBoard by running
  ```bash
  tensorboard --logdir=./tensorboard
  ```
  in the output directory (*save_dir*):

### Inference with your model
`Inference` is a part of DeepH-pack, which is used to predict the 
DFT Hamiltonian for large-scale material structures and perform 
sparse calculation of physical properties.

Firstly, one should prepare the structure file of large-scale material 
and calculate the overlap matrix. Overlap matrix calculation does not
require `SCF`. Even if the material system is large, only a small calculation
time and memory consumption are required. Following are the steps to
calculate the overlap matrix using different supported DFT packages:
1. **ABACUS**: Set the following parameters in the input file of ABACUS `INPUT`:
    ```
    calculation   get_S
    ```
    and run ABACUS like a normal `SCF` calculation.
    [ABACUS version >= 2.3.2](https://github.com/deepmodeling/abacus-develop/releases/tag/v2.3.2) is required.
2. **OpenMX**: See this [repository](https://github.com/mzjb/overlap-only-OpenMX#usage).

For overlap matrix calculation, you need to use the same basis set and DFT
software when preparing the dataset.

Then, prepare a configuration in the format of *ini*, setting up the 
file referring to the default `DeepH-pack/deeph/inference/inference_default.ini`. 
The meaning of the keywords can be found in the
[INPUT KEYWORDS section](https://deeph-pack.readthedocs.io/en/latest/keyword/inference.html). 
For a quick start, you must set up *OLP_dir*, *work_dir*, *interface*,
*trained_model_dir* and *sparse_calc_config*, as well as a `JSON` 
configuration file located at *sparse_calc_config* for sparse calculation.

With the configuration files prepared, run 
```bash
deeph-inference --config ${config_path}
```
with `${config_path}` replaced by the path of your configuration file.

## Demo: DeepH study on twisted bilayer bismuthene
When the directory structure of the code folder is not modified, the scripts in it can be used to generate a dataset of non-twisted structures, train a DeepH model, make predictions on the DFT Hamiltonian matrix of twisted structure, and perform sparse diagonalization to compute the band structure for the example study of bismuthene.

Firstly, generate example input files according to your environment path by running the following command:
```bash
cd DeepH-pack
python gen_example.py ${openmx_path} ${openmx_overlap_path} ${pot_path} ${python_interpreter} ${julia_interpreter}
```
with `${openmx_path}`, `${openmx_overlap_path}`, `${pot_path}`, `${python_interpreter}`, and `${julia_interpreter}` replaced by the path of original OpenMX executable program, modified 'overlap only' OpenMX executable program, VPS and PAO directories of OpenMX, Python interpreter, and Julia interpreter, respectively. For example, 
```bash
cd DeepH-pack
python gen_example.py /home/user/openmx/source/openmx /home/user/openmx_overlap/source/openmx /home/user/openmx/DFT_DATA19 python /home/user/julia-1.5.4/bin/julia
```

Secondly, enter the generated `example/` folder and run `run.sh` in each folder one-by-one from 1 to 5. Please note that `run.sh` should be run in the directory where the `run.sh` file is located.
```bash
cd example/1_DFT_calculation
bash run.sh
cd ../2_preprocess
bash run.sh
cd ../3_train
bash run.sh
cd ../4_compute_overlap
bash run.sh
cd ../5_inference
bash run.sh
```
The third step, the neural network training process, is recommended to be carried out on the GPU. In addition, in order to get the energy band faster, it is recommended to calculate the eigenvalues ​​of different k points in parallel in the fifth step by *which_k* interface.

After completing the calculation, you can find the band structure data in OpenMX Band format of twisted bilayer bismuthene with 244 atoms per supercell computed by the predicted DFT Hamiltonian in the file below:
```
example/work_dir/inference/5_4/openmx.Band
```
The plotted band structure will be consistent with the right pannel of figure 6c in our paper.


## Demo: Reproduce the experimental results of the paper
You can train DeepH models using the existing [dataset](https://zenodo.org/record/6555484) to reproduce the results of our paper.

Firstly, download the processed dataset for graphene (*graphene_dataset.zip*), MoS<sub>2</sub> (*MoS2_dataset.zip*), twisted bilayer graphene (*TBG_dataset.zip*) or twisted bilayer bismuthene (*TBB_dataset.zip*). Uncompress the ZIP file.

Secondly, edit corresponding config files in the `DeepH-pack/ini/`. *raw_dir* should be set to the path of the downloaded dataset. *graph_dir* and *save_dir* should be set to the path to save your graph file and results file during the training. For grahene, twisted bilayer graphene and twisted bilayer bismuthene, a single MPNN model is used for each dataset. For MoS<sub>2</sub>, four MPNN models are used. Run 
```bash
deeph-train --config ${config_path}
```
with `${config_path}` replaced by the path of config file for training.

After completing the training, you can find the trained model in *save_dir*, which can be used to make prediction on new structures by run
```bash
deeph-inference --config ${inference_config_path}
```
with `${inference_config_path}` replaced by the path of config file for inference.
