# Inference

The default value can be found in `DeepH-pack/deeph/inference/inference_default.ini`. The following arguments can be set in the configuration files for `Inference`:

## basic

- *OLP_dir* : The output directory of the 'overlap only' OpenMX calculation.

+ *work_dir* : The directory to run the workflow and save the results.

- *interface* : Which DFT package is used to get the overlap matrix. Support `abacus` and `openmx`.

+ *trained_model_dir* : The directory to the trained model. If only one model is used for the current material system, fill in the string of the directory to the trained model, e.g. `/your/trained/model`. If multiple models are used for the current material system, fill in the JSON format string containing all the directories of models, e.g. `["/your/trained/model1", "/your/trained/model2"]`.

- *task* : Set it `[1, 2, 3, 4, 5]` to run all the tasks for inference. `1` in list means to parse the overlap, `2` means to get the local coordinate, `3` means to predict the Hamiltonian, `4` means to rotate the Hamiltonian back, and `5` means to perform the sparse calculation.

+ *sparse_calc_config* : The directory to the *JSON* configuration file.

- *dense_calc* : Whether to replace sparse matrix calculation with dense matrix calculation.

+ *huge_structure* : Whether to save your memory and cost more time during inference.

- *restore_blocks_py* : Whether to use Python code to rearrange matrix blocks. You can set it `False` to use Julia code instead to improve efficiency.

## interpreter

- *julia_interpreter* : The directory to the julia interpreter.

## graph

- *radius* : The additional cut-off radius for crystal graph based on the truncation that adopted in overlap matrices. `-1.0` means using the same truncation that adopted in overlap matrices.

## *JSON* configuration file

- *calc_job* : Which quantity you want to calculate after the hamiltonian gotten. Can only be 'band' for now.

+ *fermi_level* : Fermi level.

- *k_data* : The k-path to calculate, formatted like `["number_of_points x1 y1 z1 x2 y2 z2 name_of_begin_point name_of_end_point", ...]`.

+ *which_k* : Define which point in k-path to calculate, start counting from 1. You can set it '0' for all k points, or '-1' for no point. It is recommended to calculate the eigenvalues of different k points in parallel through it. (Invalid for dense matrix calculation)

- *num_band* : The number of eigenvalues and eigenvectors desired. (Invalid for dense matrix calculation)

+ *max_iter*: Maximum number of iterations. (Invalid for dense matrix calculation)
