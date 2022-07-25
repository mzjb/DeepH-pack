# Preprocess

The default value can be found in `DeepH-pack/deeph/preprocess/preprocess_default.ini`. The following arguments can be set in the configuration file for `Preprocess`:

## basic

- *raw_dir* : The path to the root directory of your dataset. A subdirectory inside will be seen as a piece of data if there are the Hamiltonian file `openmx.scfout`.

+ *processed_dir* : The path to the root directory to save the preprocessed files. A subdirectory will be created to save the corresponding files for each piece of data. This argument can have the same value as *raw_dir*, when the preprocessed files will be created in the same directory as the corresponding `openmx.scfout` in.

- *interface* : Which DFT package is used to get the Hamiltonian. Support `openmx`, `abacus` and `aims`.

+ *multiprocessing* : Whether to use multiprocessing to perform `Preprocess` for different data.

## interpreter

- *julia_interpreter* : The path to the julia interpreter.

## graph

- *radius* : The additional cut-off radius for crystal graph based on the truncation that adopted in Hamiltonian matrices. `-1.0` means using the same truncation that adopted in Hamiltonian matrices.
