# Demo: Reproduce the experimental results of the paper
You can train DeepH models using the existing [dataset](https://zenodo.org/record/6555484) to reproduce the results of this [paper](https://www.nature.com/articles/s43588-022-00265-6).

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
Please note that the DFT results in this dataset were calculated using OpenMX.
This means that if you want to use a model trained on this dataset to calculate properties, you need to use the overlap calculated using OpenMX.
The orbital information required for overlap calculations can be found in the [paper](https://www.nature.com/articles/s43588-022-00265-6).