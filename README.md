# Learning Initial Solutions to Mixed-Integer Programs

### Overview

This release contains a working implementation of Neural Diving from the paper
[Solving Mixed Integer Programs Using Neural Networks](https://arxiv.org/abs/2012.13349)
with data generation built on top of that from [Exact Combinatorial Optimization
with Graph Convolutional Neural Networks](https://arxiv.org/abs/1906.01629).

The following gives a brief overview of the contents; more detailed
documentation is available within each file:

* __config_train.py__: Configuration file for training parameters.
* __data_generation.py__: Generates the feature set for each MIP instance.
* __data_utils.py__: Utility functions for feature extraction.
* __evaluate_solvers.py__: Compares solver performance between warm starts with predicted solutions and cold starts.
* __instance_generation.py__: Generates each MIP instance.
* __layer_norm.py__: Model layer normalisation and dropout utilities.
* __light_gnn.py__: The GNN model used for training.
* __sampling.py__: Sampling strategies for Neural LNS.
* __solution_data.py__: SolutionData classes used to log solution process.
* __solvers.py__: Neural diving and feature generation implementation.
* __train.py__: Training script for neural diving model.
* __data__: Directory with tfrecord files to run training.

## Installation

To install the dependencies of this implementation, please run:

```
conda env create -f environment.yml
conda activate neural diving
```


## Usage

1. Generate a collection of MIP instances with `instance_generation.py`.
2. Generate the features and labels to train the neural diving model with `data_generation.py`.
3. Specify valid training and validation paths in `config_train.py` (i.e.
   <dataset_absolute_training_path> and <dataset_absolute_validation_path>).
4. Train the neural diving model with `train.py`.
5. Compare warm starting Gurobi with neural diving predictions to cold starting Gurobi with `evaluate_solvers.py`.

