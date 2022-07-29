# Inference with your model

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
