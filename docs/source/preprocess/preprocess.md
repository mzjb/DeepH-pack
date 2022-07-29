# Preprocess the dataset

`Preprocess` is a part of DeepH-pack. Through `Preprocess`, 
DeepH-pack will convert the unit of physical quantity, store 
the data files in the format of text and *HDF5* for each structure 
in a separate folder, generate local coordinates, and perform basis 
transformation for DFT Hamiltonian matrices. We use the following 
convention of units:

Quantity | Unit 
---|---
Length   | Å    
Energy   | eV   

You need to edit a configuration in the format of *ini*, setting 
up the file referring to the default file 
`DeepH-pack/deeph/preprocess/preprocess_default.ini`. The meaning 
of the keywords can be found in the [INPUT KEYWORDS section](https://deeph-pack.readthedocs.io/en/latest/keyword/preprocess.html). 
For a quick start, you must set up *raw_dir*, *processed_dir* and *interface*.

With the configuration file prepared, run 
```bash
deeph-preprocess --config ${config_path}
```
with `${config_path}` replaced by the path of your configuration file.
