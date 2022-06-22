# Inference with your model

`Inference` is a part of DeepH-pack, which is used to predict the 
DFT Hamiltonian for large-scale material structures and perform 
sparse calculation of physical properties.

Firstly, one should prepare the structure file of large-scale material 
and calculate the overlap matrix with 'overlap only' OpenMX. Then,
prepare a configuration in the format of *ini*, setting up the 
file referring to the default `DeepH-pack/deeph/inference/inference_default.ini`. 
The meaning of the keywords can be found in the [INPUT KEYWORDS section](https://deeph-pack.readthedocs.io/en/latest/keyword/inference.html). 
For a quick start, you must set up *OLP_dir*, *work_dir*, 
*trained_model_dir* and *sparse_calc_config*, as well as a *JSON* 
configuration file located at *sparse_calc_config* for sparse calculation.

With the configuration files prepared, run 
```bash
deeph-inference --config ${config_path}
```
with `${config_path}` replaced by the path of your configuration file.
