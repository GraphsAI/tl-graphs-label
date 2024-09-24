# Explainable Transfer Learning on Graphs Using a Novel Label Frequency Representation

## How to reproduce
The steps below are used to reproduce the experiments of the paper.

### 1) Install required packages
Create a virtual environment (recommended):
```
python -m venv environment_name
source environment_name/bin/activate
```

In a virtual environment, run:
```
pip install -r requirements.txt
```
### 2) Create graph representation based on label frequency

The code allows the dataset to be converted from TU-Dortmund to the label frequency-based representation:

```
python convert_data.py --dataset [DATASET NAME] 
```

Optional parameters are:
- --degree: Whether to turn labels into degrees (**required for social networks**).
- --round: Decimal digit to round (*default*=4).
- --reverse: Reverse the classes of the graphs.
- --verbose.

### 3) Verify transfer learning between different graph domains 

The code for aligning representations to carry out the transfer of knowledge between different domains.

```
python classify.py --base [DATASET_BASE] --dataset [ALIGNED_DATASET_1] [ALIGNED_DATASET_2] --model [MODEL_NAME]
```

All available parameters:

- --base: Base dataset to align others
- --datasets: Other datasets usable for training
- --transfer: Joint training on all datasets
- --save_index: Save the index of alignment
- --K: Maximum number of cycles for alignment (*default*=2)
- --model: Model for classifying (*RFC, SMV, KNN, MLP*):
    - RFC: Random Forest (*default*)
    - SVM: Support Vector Machine
    - KNN: K-Nearest Neighbors
    - MLP: MultiLayer Perceptron
- --verbose.

## Example of execution for paper results

In the paper, three datasets are considered: DD, MUTAG and REDDIT-BINARY.

We convert the three datasets into the corresponding representations based on label frequency.
```
python convert_data.py --dataset DD 
python convert_data.py --dataset MUTAG --reverse 
python convert_data.py --dataset REDDIT-BINARY --degree 
```

Next, we use REDDIT-BINARY as the base while aligning DD and MUTAG. We use the Random Forest model on the joint dataset to use the knowledge also learned from DD and MUTAG for REDDIT-BINARY.

```
python classify.py --base MUTAG --datasets REDDIT-BINARY DD --model RFC --K 2 --transfer
``` 
