# ProtoEHR: Hierarchical Prototype Learning for EHR-Based Healthcare Predictions
![model](https://github.com/user-attachments/assets/9fa7e8e7-0fba-4adb-95fa-0eef7b55a2d0)
```bibtex
@inproceddings{cai2025protoehr,
	title 	  = {ProtoEHR: Hierarchical Prototype Learning for EHR-based Healthcare Predictions},
	author	  = {Cai, Zi and Liu, Yu and Luo, Zhiyao and Zhu, Tingting},
	booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
	year      = {2025},
}
```

## Requirements
Start by configuring the environment
```bash
cd ./ProtoEHR
conda env create -n ProtoEHR -f environment.yaml
conda activate ProtoEHR
```
Acquire the required MIMIC-III and MIMIC-IV datasets from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)<br>
<br>
For phenotype prediction, our ./data/{DATASET_NAME}/ccs_phenotypes.pickle are extracted from hcup_ccs_2015_definitions.yaml proposed in [Multitask learning and benchmarking with clinical time series data](https://github.com/YerevaNN/mimic3-benchmarks/tree/master)<br>
<br>
Initialize all directories
```bash
python create_project_structure.py
```

## Data Preparation
### 1. Generate LLM Graph
```bash
python ./gen_llm_relations/main.py
```
Save the LLM graph file under ./data/processed/{DATASET_NAME}

### 2. Create Datasets
Specify the following arguments in ./configs/data_preparation.yaml to the desired values
```yaml
dataset: mimiciii                              # or mimiciv
task: data_preparation-mortality_prediction    # data_preparation-{TASK_NAME}

dataset_filtering:
  args:
    code_thresh: 100                           # 60 for mimiciv
```
Create the training, validation, and testing files
```bash
python ./data_preparation/data_preparation.yaml -c ./configs/data_preparation.yaml
```

## Train and Evaluate ProtoEHR
Runnning ProtoEHR requires first setting up the corresponding configs file to specify the dataset, task, or model parameters
```yaml
dataset: mimiciii                              # or mimiciv
task: mortality_prediction    
code_thresh: 100  # 60 for mimiciv

model:
  name: ProtoEHR
  model_type: base
  mode: train
  freeze: False
  args:

code_proto_num: 32
visit_proto_num: 16
patient_proto_num: 8
...
```
By subsequently running the main file, arguments in the config file will be used for training and evaluation. Our training procedure uses early stopping to terminate the process, after termination, evaluation on the test set is performed via bootstrapping.
```bash
python ./main.py -c ./configs/base.yaml
```

## Train and Evaluate Baselines
Our proposed framework is evaluated against the following baselines: Deepr, GRASP, AdaCare, StageNet, GraphCare, and KerPrint. A similar procedure is followed to configure, train, and evaluate the baselines. To run the main file with the specified baseline model, substitute {BASELINE_NAME} with one of the names above.
```bash
python ./main.py -c ./configs/{BASELINE_NAME}.yaml
```
