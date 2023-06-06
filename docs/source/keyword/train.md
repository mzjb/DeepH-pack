# Train

The default value can be found in `DeepH-pack/deeph/default.ini`. The following arguments can be set in the configuration file for `Train`:

## basic

- *graph_dir* : The directory to save the graph for the dataset.

+ *save_dir* : The root directory to save the training result.

- *raw_dir* : The root directory of the preprocessed dataset.

+ *dataset_name* : The name of your dataset.

- *disable_cuda* : Whether to disable the cuda during training.

+ *device* : The device you used for training (`cpu` or `cuda:x`, where `x` is the index of the cuda device). If the cuda is disabled by *disable_cuda* or it is not available in your environment, you may not set this option as it will be automatically set as 'cpu'.

- *num_threads* : The number of threads used for PyTorch on CPU.

+ *save_to_time_folder* : Whether to create a subfolder named with the current time in *save_dir*.

- *save_csv* : Whether to output labels and predictions for all the structures in the format of csv.

+ *tb_writer* : Whether to track and visualize the training process by TensorBoard.

- *seed* : The seed for generating random numbers.

+ *multiprocessing* : Number of processes to use multiprocessing to generate crystal graphs. Set to `-1` to use all available CPUs. Set to `0` (default) to disable multiprocessing. WARNING: The keyword "num_threads" is incompatible with `multiprocessing`. If you use `multiprocessing` with a value of 1 or higher, the crystal graphs generation process will ignore the `num_threads` keyword. For optimal performance and memory usage, we recommend setting `multiprocessing = 0` and adjusting `num_threads` to control the number of threads. This is because generating crystal graphs can consume large memory. 

- *orbital* : A JSON format string that defines matrix elements to be predicted. For example, let ![](https://latex.codecogs.com/svg.image?H_{i,&space;\alpha}^{j,&space;\beta}) denotes DFT Hamiltonian matrix element between orbital ![](https://latex.codecogs.com/svg.image?\alpha) of atom ![](https://latex.codecogs.com/svg.image?i) and orbital ![](https://latex.codecogs.com/svg.image?\beta) of atom ![](https://latex.codecogs.com/svg.image?j). An input of `[{"N1 N2": [a1, a2], "N3 N4": [a3, a4], "N5 N6": [a5, a6]}, {"N7 N8": [a7, a8]}]` can be set for the *orbital* option, if you want to predict two matrix elements `H1` and `H2` for the edge feature of an atom pair ![](https://latex.codecogs.com/svg.image?ij), where <div align=center><img
 src="https://latex.codecogs.com/svg.image?\textsf{H1}=\left\{\begin{aligned}H_{i,&space;\textsf{a1}}^{j,&space;\textsf{a2}}&space;&\textsf{,&space;if&space;the&space;atomic&space;number&space;of&space;}&space;i&space;\textsf{&space;and&space;}&space;j&space;\textsf{&space;is&space;N1&space;and&space;N2,&space;respectively}\\H_{i,&space;\textsf{a3}}^{j,&space;\textsf{a4}}&space;&\textsf{,&space;if&space;the&space;atomic&space;number&space;of&space;}&space;i&space;\textsf{&space;and&space;}&space;j&space;\textsf{&space;is&space;N3&space;and&space;N4,&space;respectively}\\H_{i,&space;\textsf{a5}}^{j,&space;\textsf{a6}}&space;&\textsf{,&space;if&space;the&space;atomic&space;number&space;of&space;}&space;i&space;\textsf{&space;and&space;}&space;j&space;\textsf{&space;is&space;N5&space;and&space;N6,&space;respectively}\\\textsf{None}&space;&\textsf{,&space;otherwise}\\\end{aligned}\right."
 class="aligncenter"
 title="
 \textsf{H1}=\left\{
 \begin{aligned}
 H_{i, \textsf{a1}}^{j, \textsf{a2}} &\textsf{, if the atomic number of } i \textsf{ and } j \textsf{ is N1 and N2, respectively}\\
 H_{i, \textsf{a3}}^{j, \textsf{a4}} &\textsf{, if the atomic number of } i \textsf{ and } j \textsf{ is N3 and N4, respectively}\\
 H_{i, \textsf{a5}}^{j, \textsf{a6}} &\textsf{, if the atomic number of } i \textsf{ and } j \textsf{ is N5 and N6, respectively}\\
 \textsf{None} &\textsf{, otherwise}\\
 \end{aligned}
 \right.
 "
/></div> and <div align=center><img
 src="https://latex.codecogs.com/svg.image?\textsf{H2}=\left\{\begin{aligned}H_{i,&space;\textsf{a7}}^{j,&space;\textsf{a8}}&space;&\textsf{,&space;if&space;the&space;atomic&space;number&space;of&space;}&space;i&space;\textsf{&space;and&space;}&space;j&space;\textsf{&space;is&space;N7&space;and&space;N8,&space;respectively}\\\textsf{None}&space;&\textsf{,&space;otherwise}\\\end{aligned}\right."
 title="
 \textsf{H2}=\left\{
 \begin{aligned}
 H_{i, \textsf{a7}}^{j, \textsf{a8}} &\textsf{, if the atomic number of} i \textsf{and} j \textsf{is N7 and N8, respectively}\\
 \textsf{None} &\textsf{, otherwise}\\
 \end{aligned}
 \right.
 "
/></div> Alternatively, a Python script at `DeepH-pack/tools/get_all_orbital_str.py` can be used to generate a default configuration to predict all orbitals with one model.

## graph

- *create_from_DFT* : Whether to use the DFT Hamiltonian matrices to create the graph instead of setting the cut-off radius by hand. It is recommended to set *create_from_DFT* to `True` and not to set *radius* for training.

+ *radius* : The cut-off radius to create graph. Keyword *radius* has no effect if *create_from_DFT* is set to `True`.

## train

- *epochs* : The number of passes of the entire training dataset the  learning algorithm has completed.

+ *pretrained* : The path to the pretrained model, e.g. `/your/pretrained/model/best_state_dict.pkl`.

- *resume* : The path to the half-trained model, e.g. `/your/half_trained/model/best_state_dict.pkl`.

+ *train_ratio* : The ratio of training data.

- *val_ratio* : The ratio of validation data.

+ *test_ratio* : The ratio of test data.

## hyperparameter

- *batch_size* : The size of mini-batch.

+ *learning_rate* : Initial learning rate.

## network

- *atom_fea_len* : The number of atom features in MPNN layers.

+ *edge_fea_len* : The number of edge features in MPNN layers.

- *gauss_stop* : The stopping radius of basis functions used to represent interatomic distances.

+ *num_l* : The number of angular quantum numbers that spherical harmonic functions have.

- *distance_expansion* : Which basis functions are used to represent interatomic distances. `choices = ['GaussianBasis', 'BesselBasis', 'ExpBernsteinBasis']`

+ *normalization* : Which form of normalization layers are used. `choices = ['BatchNorm', 'LayerNorm', 'PairNorm', 'InstanceNorm', 'GraphNorm', 'DiffGroupNorm', 'None']`

-*atom_update_net* : Which form of convolutional layers to update atom features are used. `choices = ['CGConv', 'GAT', 'PAINN']`

