# Design-Bench

Design-Bench is a **benchmarking framework** for solving automatic design problems that involve choosing an input that maximizes a black-box function. This type of optimization is used across scientific and engineering disciplines in ways such as designing proteins and DNA sequences with particular functions, chemical formulas and molecule substructures, the morphology and controllers of robots, and many more applications. 

These applications have significant potential to accelerate research in biochemistry, chemical engineering, materials science, robotics and many other disciplines. We hope this framework serves as a robust platform to drive these applications and create widespread excitement for model-based optimization.

## Offline Model-Based Optimization

![Offline Model-Based Optimization](https://design-bench.s3-us-west-1.amazonaws.com/mbo.png)

The goal of model-based optimization is to find an input **x** that maximizes an unknown black-box function **f**. This function is frequently difficulty or costly to evaluate---such as requiring wet-lab experiments in the case of protein design. In these cases, **f** is described by a set of function evaluations: D = {(x_0, y_0), (x_1, y_1), ... (x_n, y_n)}, and optimization is performed without querying **f** on new data points.

## Installation

Design-Bench can be installed with the complete set of benchmarks via our pip package.

```bash
pip install design-bench[all]
```

Alternatively, if you do not have MuJoCo, you may opt for a minimal install.

```bash
pip install design-bench
```

## Available Tasks

In the below table, we list the supported datasets and objective functions for model-based optimization, where a :heavy_check_mark: indicates that a particular combination has been tested and is available for download from our server.

Dataset \ Oracle | Exact | Gaussian Process | Random Forest | Fully Connected | LSTM | ResNet | Transformer
---------------- | ----- | ---------------- | ------------- | --------------- | ---- | --- | -----------
TF Bind 8 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
GFP |  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
ChEMBL |  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
UTR |  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
HopperController | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |  | 
Superconductor |  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |  | 
TF Bind 10 | :heavy_check_mark: |  |  |  |  |  | 
NAS Bench 101 | :heavy_check_mark: |  |  |  |  |  | 

Combinations of datasets and oracles that are not available for download from our server are automatically trained on your machine on task creation. This currently only affects approximate oracles on TF Bind 10 and NAS Bench 101. Below we provide the preferred oracle for each task, as well as meta data such as the number of data points measured.

Task Name | Dataset | Oracle | Dataset Size | Spearman's ρ
--------- | ------- | ------ | ------------ | ----------------
TFBind8-Exact-v0 | TF Bind 8 | Exact | 65792 | 
GFP-Transformer-v0 | GFP | Transformer | 56086 | 0.8497
ChEMBL-ResNet-v0 | ChEMBL | ResNet | 40516 | 0.3208
UTR-Transformer-v0 | UTR | Transformer | 560000 | 0.6425
HopperController-Exact-v0 | Hopper Controller | Exact | 3200 | 
Superconductor-FullyConnected-v0 | Superconductor | Fully Connected | 21263 | 0.9210
TFBind10-Exact-v0 | TF Bind 10 | Exact | 8321066 | 
NASBench-Exact-v0 | NAS Bench 101 | Exact | 1293208 | 

## Task API

Design-Bench tasks share a common interface specified in **design_bench/task.py**, which exposes a set of input designs **task.x** and a set of output predictions **task.y**. In addition, the performance of a new set of input designs (such as those output from a model-based optimization algorithm) can be found using **y = task.predict(x)**.

```python
import design_bench
task = design_bench.make('TFBind8-Exact-v0')

def optimize(x0, y0):
    return x0  # solve a model-based optimization problem

# solve for the best input x_star and evaluate it
x_star = optimize(task.x, task.y)
y_star = task.predict(x_star)
```

Many datasets of interest to practitioners are too large to load in memory all at once, and so the task interface defines an several iterables that load samples from the dataset incrementally.
 
 ```python
import design_bench
task = design_bench.make('TFBind8-Exact-v0')

for x, y in task:
    pass  # train a model here
    
for x, y in task.iterate_batches(32):
    pass  # train a model here
    
for x, y in task.iterate_samples():
    pass  # train a model here
 ```
 
Certain optimization algorithms require a particular input format, and so tasks support normalization of both **task.x** and **task.y**, as well as conversion of **task.x** from discrete tokens to the logits of a categorical probability distribution---needed when optimizing **x** with a gradient-based model-based optimization algorithm.
 
 ```python
import design_bench
task = design_bench.make('TFBind8-Exact-v0')

# convert x to logits of a categorical probability distribution
task.map_to_logits()
discrete_x = task.to_integers(task.x)

# normalize the inputs to have zero mean and unit variance
task.map_normalize_x()
original_x = task.denormalize_x(task.x)

# normalize the outputs to have zero mean and unit variance
task.map_normalize_y()
original_y = task.denormalize_y(task.y)
 ```
 
Each task provides access to the model-based optimization dataset used to learn the oracle (where applicable) as well as the oracle itself, which includes metadata for how it was trained (where applicable). These provide fine-grain control over the data distribution for model-based optimization.
 
 ```python
import design_bench
task = design_bench.make('GFP-GP-v0')

# an instance of the DatasetBuilder class from design_bench.datasets.dataset_builder
dataset = task.dataset

# modify the distribution of the task dataset
dataset.subsample(max_samples=10000, 
                   min_percentile=10,
                   max_percentile=90)

# an instance of the OracleBuilder class from design_bench.oracles.oracle_builder
oracle = task.oracle

# check how the model was fit
print(oracle.params["rank_correlation"],
       oracle.params["model_kwargs"],
       oracle.params["split_kwargs"])
 ```
