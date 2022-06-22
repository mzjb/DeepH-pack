# Inference

The default value can be found in `DeepH-pack/deeph/inference/inference_default.ini`. The following arguments can be set in the configuration files for `Inference`:

## basic

- *OLP_dir* : The output directory of the 'overlap only' OpenMX calculation.

+ *work_dir* : The directory to run the workflow and save the results.

- *trained_model_dir* : The directory to the trained model. If only one model is used for the current material system, fill in the string of the directory to the trained model. If multiple models are used for the current material system, fill in the JSON format string containing all the directories of models, e.g. `["/your/trained/model1", "/your/trained/model2"]`.

+ *task* : Set it `[1, 2, 3, 4, 5]` to run all the tasks for inference. `1` in list means to parse the overlap, `2` means to get the local coordinate, `3` means to predict the Hamiltonian, `4` means to rotate the Hamiltonian back, and `5` means to perform the sparse calculation.

- *sparse_calc_config* : The directory to the *JSON* configuration file.

+ *huge_structure* : Whether to save your memory and cost more time during inference.

- *restore_blocks_py* : Whether to use Python code to rearrange matrix blocks. You can set it `False` to use Julia code instead to improve efficiency.

## interpreter

- *julia_interpreter* : The directory to the julia interpreter.

## graph

- *radius* : The cut-off radius to create the local coordinate systems.

## *JSON* configuration file

- *calc_job* : Which quantity you want to calculate after the hamiltonian gotten. Can only be 'band' for now.

+ *which_k* : Define whick point in k-path to calculate, start counting from 1. You can set it '0' for all k points, or '-1' for no point. It is recommended to calculate the eigenvalues of different k points in parallel through it.

- *fermi_level* : Fermi level.

+ *lowest_band* : Find eigenvalues above *lowest_band* using shift-invert mode.

- *max_iter*: Maximum number of iterations.

+ *num_band* : The number of eigenvalues and eigenvectors desired.

- *k_data* : The k-path to calculate, formatted like `["number_of_points x1 y1 z1 x2 y2 z2 name_of_begin_point name_of_end_point", ...]`.