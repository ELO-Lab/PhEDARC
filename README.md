# Stable and Sample-Efficient Policy Search for Continuous Control via Hybridizing Phenotypic Evolutionary Algorithm with the Double Actors Regularized Critics
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Thai Huy Nguyen, Ngoc Hoang Luong

<!-- In Applied Intelligence 2021. -->
Official PyTorch implementation of **PhEDARC**
## Setup
- Clone this repo
```
git clone https://github.com/NtHuy07/PhEDARC.git
```
You should have mujoco-py on your running device, which could be installed from [here](https://github.com/openai/mujoco-py). 
- Install dependencies using `conda`:
```
conda env create -f environment.yaml
conda activate phedarc
```
## Instruction
### Train agent with PhEDARC
- To reproduce results in the paper
```shell
python run_exp.py --env <environment_name> 
		  --distil --phenotypic_mut --stable
	  	  --logdir <logging_directory> --seed <seed_number>
```
- In addition, you could replace the default mutation operator with proximal mutation
```shell
python run_exp.py --env <environment_name>
		  --distil --proximal_mut --stable
		  --logdir <logging_directory> --seed <seed_number>
```
### Other options
|Hyperparameters          |Help                                                    |Default           |                
|:------------------------|:-------------------------------------------------------|:-----------------|
|`--save_periodic`        |Save actor, critic and memory periodically              |N/A               |
|`--save_csv_freq`        |Frequency (in generations) to store score statistics    |1                 |
|`--next_frame_save`      |Actor save frequency (by frames)                        |1000000           |
|`--num_test_evals`       |Number of episodes to evaluate test score               |5                 |

***Note:*** You could find all the hyperparameters settings in the parameters.py file.
### Evaluate & Rendering
```shell
python evaluate.py --env <environment_name> --model_path <path_to_actor_model> --render 
```
## Acknowledgements
Our source code is heavily relied and inspired by:
- [Proximal Distilled Evolutionary Reinforcement Learning](https://github.com/crisbodnar/pderl)
- [Efficient Continuous Control with Double Actors and Regularized Critics](https://github.com/dmksjfl/DARC)
