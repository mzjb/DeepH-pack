# Demo: DeepH study on twisted bilayer bismuthene
When the directory structure of the code folder is not modified, the scripts in it can be used to generate a dataset of non-twisted structures, train a DeepH model, make predictions on the DFT Hamiltonian matrix of twisted structure, and perform sparse diagonalization to compute the band structure for the example study of bismuthene.

Firstly, generate example input files according to your environment path by running the following command:
```bash
cd DeepH-pack
python gen_example.py ${openmx_path} ${openmx_overlap_path} ${pot_path} ${python_interpreter} ${julia_interpreter}
```
with `${openmx_path}`, `${openmx_overlap_path}`, `${pot_path}`, `${python_interpreter}`, and `${julia_interpreter}` replaced by the path of original OpenMX executable program, modified 'overlap only' OpenMX executable program, VPS and PAO directories of OpenMX, python interpreter, and julia interpreter, respectively. For example, 
```bash
cd DeepH-pack
python gen_example.py /home/user/openmx/source/openmx /home/user/openmx_overlap/source/openmx /home/user/openmx/DFT_DATA19 python /home/user/julia-1.5.4/bin/julia
```

Secondly, enter the generated `example/` folder and run `run.sh` in each folder one-by-one from 1 to 5. Please note that `run.sh` should be run in the directory where the `run.sh` file is located.
```bash
cd example/1_DFT_calculation
bash run.sh
cd ../2_preprocess
bash run.sh
cd ../3_train
bash run.sh
cd ../4_compute_overlap
bash run.sh
cd ../5_inference
bash run.sh
```
The third step, the neural network training process, is recommended to be carried out on the GPU. In addition, in order to get the energy band faster, it is recommended to calculate the eigenvalues ​​of different k points in parallel in the fifth step by *which_k* interface.

After completing the calculation, you can find the band structure data in OpenMX Band format of twisted bilayer bismuthene with 244 atoms per supercell computed by the predicted DFT Hamiltonian in the file below:
```
example/work_dir/inference/5_4/openmx.Band
```
The plotted band structure will be consistent with the right pannel of figure 6c in our paper.
