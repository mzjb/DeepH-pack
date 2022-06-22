# Preprocess

The default value can be found in `DeepH-pack/deeph/preprocess/preprocess_default.ini`. The following arguments can be set in the configuration file for `Preprocess`:

## basic

- *raw_dir* : The path to the root directory of your dataset. A subdirectory inside will be seen as a piece of data if there are the Hamiltonian file `openmx.scfout`.

+ *processed_dir* : The path to the root directory to save the preprocessed files. A subdirectory will be created to save the corresponding files for each piece of data. This argument can have the same value as *raw_dir*, when the preprocessed files will be created in the same directory as the corresponding `openmx.scfout` in.

- *interface* : Which DFT package is used to get the Hamiltonian. Can only be 'openmx' for now.

## interpreter

- *julia_interpreter* : The path to the julia interpreter.

## graph

- *radius* : The cut-off radius to create the local coordinate systems. It is required bigger than the cut-off radius of local basis set.