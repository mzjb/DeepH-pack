# Train your model

`Train` is a part of DeepH-pack, which is used to train a deep 
learning model using the processed dataset.

Prepare a configuration in the format of *ini*, setting up the 
file referring to the default `DeepH-pack/deeph/default.ini`. 
The meaning of the keywords can be found in the [INPUT KEYWORDS section](https://deeph-pack.readthedocs.io/en/latest/keyword/train.html). 
For a quick start, you must set up *graph_dir*, *save_dir*, 
*raw_dir* and *orbital*, other keywords can stay default and 
be adjusted later.

With the configuration file prepared, run 
```bash
deeph-train --config ${config_path}
```
with `${config_path}` replaced by the path of your configuration file.

Tips:
- **Name your dataset**. Use *dataset_name* to name your dataset, 
the same names may overwrite each other.

- **Hyperparameters of the neural network**. The neural network here contains 
some hyperparameters. For a specific problem your should try adjusting 
the hyperparameters to obtain better results.

- **The keyword *orbital***. The keyword *orbital* states which orbitals or matrix elements are predicted. It is a little complicated to understand its data structure. To figure out it, you can refer to the [INPUT KEYWORDS section](https://deeph-pack.readthedocs.io/en/latest/keyword/train.html#:~:text=generate%20crystal%20graphs.-,orbital,-%3A%20A%20JSON%20format) or the method [make_mask](https://github.com/mzjb/DeepH-pack/blob/main/deeph/kernel.py#:~:text=def%20make_mask(self%2C%20dataset)%3A) in class `DeepHKernal` defined in `DeepH-pack/deeph/kernel.py`.

    Alternatively, a Python script at `DeepH-pack/tools/get_all_orbital_str.py` can be used to generate a default configuration to predict all orbitals with one model.

- **Use TensorBoard for visualizations**. You can track and visualize the training process through TensorBoard by running
  ```bash
  tensorboard --logdir=./tensorboard
  ```
  in the output directory (*save_dir*):