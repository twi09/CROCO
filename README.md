
# CROCO - COst-efficient RObust COunterfactuals 

CROCO is a method for generating robust counterfactuals in the context of recourse invalidation rate.
Our library is based on the CARLA framework: https://github.com/carla-recourse/CARLA

<img src="cool_croco.png" alt="Screenshot" width="20%" height="20%">


### Datasets

- Adult Data Set: [Source](https://archive.ics.uci.edu/ml/datasets/adult)
- COMPAS: [Source](https://www.kaggle.com/danofer/compass)
- Give Me Some Credit (GMC): [Source](https://www.kaggle.com/c/GiveMeSomeCredit/data)

### Evaluated counterfactual methods

- Wachter: [Paper](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
- PROBE: [Paper](https://openreview.net/forum?id=sC-PmTsiTB)


### Used machine learning model

- **ANN**: Artificial Neural Network with 2 hidden layers and ReLU activation function



## Installation

### Requirements

- `python3.7`
- `pip`


### Install python 3.7 env with conda 
```
conda create -n carla_test python=3.7
```
### Install the carla-recourse library (that contains CROCO)

```
pip install -e . 
```

## Reproduct paper results 


### Run the experiments 
```
python3 run_experiments.py 
```
### Format the results for analysis 
```
python3 export_results.py 
```

### Create the plots in R 
```
Rscript graphics.R
```

