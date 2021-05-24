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
pip install morphing-agents==1.4
```

Alternatively, if you do not have MuJoCo, you may opt for a minimal install.

```bash
pip install design-bench
```

## Available Tasks

In the below table, we list the supported datasets and objective functions for model-based optimization, where a ✅ indicates that a particular combination has been tested and is available for download from our server.

Dataset \ Oracle | Exact | Gaussian Process | Random Forest | Fully Connected | LSTM | ResNet | Transformer
---------------- | ----- | ---------------- | ------------- | --------------- | ---- | --- | -----------
TF Bind 8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅
GFP |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅
ChEMBL |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅
UTR |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅
Hopper Controller | ✅ | ✅ | ✅ | ✅ |  |  | 
Superconductor |  | ✅ | ✅ | ✅ |  |  | 
Ant Morphology | ✅ | ✅ | ✅ | ✅ |  |  | 
D'Kitty Morphology | ✅ | ✅ | ✅ | ✅ |  |  | 

Combinations of datasets and oracles that are not available for download from our server are automatically trained on your machine on task creation. This currently only affects approximate oracles on user-defined MBO tasks. Below we provide the preferred oracle for each task, as well as meta data such as the number of data points measured.

Task Name | Dataset | Oracle | Dataset Size | Spearman's ρ
--------- | ------- | ------ | ------------ | ----------------
TFBind8-Exact-v0 | TF Bind 8 | Exact | 65792 | 
GFP-Transformer-v0 | GFP | Transformer | 56086 | 0.8497
ChEMBL-ResNet-v0 | ChEMBL | ResNet | 40516 | 0.3208
UTR-Transformer-v0 | UTR | Transformer | 560000 | 0.6425
HopperController-Exact-v0 | Hopper Controller | Exact | 3200 | 
Superconductor-FullyConnected-v0 | Superconductor | Fully Connected | 21263 | 0.9210
AntMorphology-Exact-v0 | Ant Morphology | Exact | 12300 | 
DKittyMorphology-Exact-v0 | D'Kitty Morphology | Exact | 9546 | 

## Task API

Design-Bench tasks share a common interface specified in **design_bench/task.py**, which exposes a set of input designs **task.x** and a set of output predictions **task.y**. In addition, the performance of a new set of input designs (such as those output from a model-based optimization algorithm) can be found using **y = task.predict(x)**.

```python
import design_bench
task = design_bench.make('TFBind8-Exact-v0')

def solve_optimization_problem(x0, y0):
    return x0  # solve a model-based optimization problem

# solve for the best input x_star and evaluate it
x_star = solve_optimization_problem(task.x, task.y)
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

# remove the normalization applied to the outputs
task.map_denormalize_y()
normalized_y = task.normalize_y(task.y)

# remove the normalization applied to the inputs
task.map_denormalize_x()
normalized_x = task.normalize_x(task.x)

# convert x back to integers
task.map_to_integers()
continuous_x = task.to_logits(task.x)
 ```
 
Each task provides access to the model-based optimization dataset used to learn the oracle (where applicable) as well as the oracle itself, which includes metadata for how it was trained (where applicable). These provide fine-grain control over the data distribution for model-based optimization.
 
 ```python
import design_bench
task = design_bench.make('GFP-GP-v0')

# an instance of the DatasetBuilder class from design_bench.datasets.dataset_builder
dataset = task.dataset

# modify the distribution of the task dataset
dataset.subsample(max_samples=10000, min_percentile=10, max_percentile=90)

# an instance of the OracleBuilder class from design_bench.oracles.oracle_builder
oracle = task.oracle

# check how the model was fit
print(oracle.params["rank_correlation"],
       oracle.params["model_kwargs"],
       oracle.params["split_kwargs"])
 ```

## Dataset API

Datasets provide a model-based optimization algorithm with information about the black-box function, and are used in design bench to fit approximate oracle models when an exact oracle is not available. All datasets inherit from the DatasetBuilder class defined in *design_bench.datasets.dataset_builder*.

All datasets implement methods for modifying the format and distribution of the dataset, including normalization, subsampling, relabelling the outputs, and (for discrete datasets) converting discrete inputs to real-valued. There are also special methods for splitting the dataset into a training and validation set.

<details>

<summary>Display code snippet</summary>

```python
from design_bench.datasets.discrete import GFPDataset
dataset = GFPDataset()

# convert x to logits of a categorical probability distribution
dataset.map_to_logits()
discrete_x = dataset.to_integers(dataset.x)

# normalize the inputs to have zero mean and unit variance
dataset.map_normalize_x()
original_x = dataset.denormalize_x(dataset.x)

# normalize the outputs to have zero mean and unit variance
dataset.map_normalize_y()
original_y = dataset.denormalize_y(dataset.y)

# remove the normalization applied to the outputs
dataset.map_denormalize_y()
normalized_y = dataset.normalize_y(dataset.y)

# remove the normalization applied to the inputs
dataset.map_denormalize_x()
normalized_x = dataset.normalize_x(dataset.x)

# convert x back to integers
dataset.map_to_integers()
continuous_x = dataset.to_logits(dataset.x)

# modify the distribution of the dataset
dataset.subsample(max_samples=10000, min_percentile=10, max_percentile=90)

# change the outputs as a function of their old values
dataset.relabel(lambda x, y: y ** 2 - 2.0 * y)

# split the dataset into a validation set
training, validation = dataset.split(val_fraction=0.1)
```

</details>

If you would like to define your own dataset for use with design-bench, you can directly instantiate a continuous dataset or a discrete dataset depending on the input format you are using. The DiscreteDataset class and ContinuousDataset are built with this in mind, and accept both two numpy arrays containing inputs *x* outputs *y*.

<details>

<summary>Display code snippet</summary>

```python
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.datasets.continuous_dataset import ContinuousDataset
import numpy as np

# create dummy inputs and outputs for model-based optimization
x = np.random.randint(500, size=(5000, 43))
y = np.random.uniform(size=(5000, 1))

# create a discrete dataset for those inputs and outputs
dataset = DiscreteDataset(x, y)

# create dummy inputs and outputs for model-based optimization
x = np.random.uniform(size=(5000, 871))
y = np.random.uniform(size=(5000, 1))

# create a continuous dataset for those inputs and outputs
dataset = ContinuousDataset(x, y)
```

</details>

In the event that you are using a dataset that is saved to a set of sharded numpy files (ending in .npy), you may also create dataset by providing a list of shard files representing using the DiskResource class. The DiscreteDataset class and ContinuousDataset accept two lists of sharded inputs *x* and outputs *y* represented by DiskResource objects.

<details>

<summary>Display code snippet</summary>

```python
from design_bench.disk_resource import DiskResource
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.datasets.continuous_dataset import ContinuousDataset
import os
import numpy as np

# list the disk resource for each shard
os.makedirs("new_dataset/")
x = [DiskResource("new_dataset/shard-x-0.npy"), 
     DiskResource("new_dataset/shard-x-1.npy")]
y = [DiskResource("new_dataset/shard-y-0.npy"), 
     DiskResource("new_dataset/shard-y-1.npy")]

# create dummy inputs and outputs for model-based optimization
xs = np.random.randint(500, size=(5000, 43))
ys = np.random.uniform(size=(5000, 1))

# save the dataset to a set of shard files
np.save("new_dataset/shard-x-0.npy", xs[:3000])
np.save("new_dataset/shard-x-1.npy", xs[3000:])
np.save("new_dataset/shard-y-0.npy", ys[:3000])
np.save("new_dataset/shard-y-1.npy", ys[3000:])

# create a discrete dataset for those inputs and outputs
dataset = DiscreteDataset(x, y)

# create dummy inputs and outputs for model-based optimization
xs = np.random.uniform(size=(5000, 871))
ys = np.random.uniform(size=(5000, 1))

# save the dataset to a set of shard files
np.save("new_dataset/shard-x-0.npy", xs[:3000])
np.save("new_dataset/shard-x-1.npy", xs[3000:])
np.save("new_dataset/shard-y-0.npy", ys[:3000])
np.save("new_dataset/shard-y-1.npy", ys[3000:])

# create a continuous dataset for those inputs and outputs
dataset = ContinuousDataset(x, y)
```

</details>

## Oracle API

Oracles provide a way of measuring the performance of candidate solutions to a model-based optimization problem, found by a model-based optimization algorithm, without having to perform additional real-world experiments. To this end, oracle implement a prediction function **oracle.predict(x)** that takes a set of designs and makes a prediction about their performance. The goal of model-based optimization is to maximize the predictions of the oracle. 

<details>

<summary>Display code snippet</summary>

```python
from design_bench.datasets.discrete import GFPDataset
from design_bench.oracles.tensorflow import TransformerOracle

# create a dataset and a noisy oracle
dataset = GFPDataset()
oracle = TransformerOracle(dataset, noise_std=0.1)

def solve_optimization_problem(x0, y0):
    return x0  # solve a model-based optimization problem

# evaluate the performance of the solution x_star
x_star = solve_optimization_problem(dataset.x, dataset.y)
y_star = oracle.predict(x_star)
```

</details>

In order to handle when an exact ground truth is unknown or not tractable to evaluate, Design-Bench provides a set of approximate oracles including a Gaussian Process, Random Forest, and several deep neural network architectures specialized to particular data modalities. These approximate oracles may have the following parameters.

<details>

<summary>Display code snippet</summary>

```python
from design_bench.datasets.discrete import GFPDataset
from design_bench.oracles.tensorflow import TransformerOracle

# parameters for the transformer architecture
model_kwargs=dict(
    hidden_size=64,
    feed_forward_size=256,
    activation='relu',
    num_heads=2,
    num_blocks=4,
    epochs=20,
    shuffle_buffer=60000,
    learning_rate=0.0001,
    dropout_rate=0.1)

# parameters for building the validation set
split_kwargs=dict(
    val_fraction=0.1,
    subset=None,
    shard_size=5000,
    to_disk=True,
    disk_target="gfp/split",
    is_absolute=False)
    
# create a transformer oracle for the GFP dataset
dataset = GFPDataset()
oracle = TransformerOracle(
    dataset, 
    noise_std=0.0,
    
    # parameters for ApproximateOracle subclasses
    disk_target="new_model.zip",
    is_absolute=True,
    fit=True,
    max_samples=None,
    max_percentile=100,
    min_percentile=0,
    model_kwargs=model_kwargs,
    split_kwargs=split_kwargs)

def solve_optimization_problem(x0, y0):
    return x0  # solve a model-based optimization problem

# evaluate the performance of the solution x_star
x_star = solve_optimization_problem(dataset.x, dataset.y)
y_star = oracle.predict(x_star)
```

</details>

## Defining New MBO Tasks

New model-based optimization tasks are simple to create and register with design-bench. By subclassing either DiscreteDataset or ContinuousDataset, and providing either a pair of numpy arrays containing inputs and outputs, or a pair of lists of DiskResource shards containing inputs and outputs, you can define your own model-based optimization dataset class. Once a custom dataset class is created, you can register it as a model-based optimization task by choosing an appropriate oracle type, and making a call to the register function. After doing so, subsequent calls to **design_bench.make** can find your newly registered model-based optimization task.

<details>

<summary>Display code snippet</summary>

```python
from design_bench.datasets.continuous_dataset import ContinuousDataset
import design_bench
import numpy as np

# define a custom dataset subclass of ContinuousDataset
class QuadraticDataset(ContinuousDataset):

    def __init__(self, **kwargs):
    
        # define a set of inputs and outputs of a quadratic function
        x = np.random.normal(0.0, 1.0, (5000, 7))
        y = (x ** 2).sum(keepdims=True)
        
        # pass inputs and outputs to the base class
        super(QuadraticDataset, self).__init__(x, y, **kwargs)

# parameters used for building the validation set
split_kwargs=dict(
    val_fraction=0.1,
    subset=None,
    shard_size=5000,
    to_disk=True,
    disk_target="quadratic/split",
    is_absolute=True)

# parameters used for building the model
model_kwargs=dict(
    hidden_size=512,
    activation='relu',
    num_layers=2,
    epochs=5,
    shuffle_buffer=5000,
    learning_rate=0.001)

# keyword arguments for building the dataset
dataset_kwargs=dict(
    max_samples=None,
    max_percentile=80,
    min_percentile=0)

# keyword arguments for training FullyConnected oracle
oracle_kwargs=dict(
    noise_std=0.0,
    max_samples=None,
    max_percentile=100,
    min_percentile=0,
    split_kwargs=split_kwargs,
    model_kwargs=model_kwargs)

# register the new dataset with design_bench
design_bench.register(
    'Quadratic-FullyConnected-v0', QuadraticDataset,
    'design_bench.oracles.tensorflow:FullyConnectedOracle',
    dataset_kwargs=dataset_kwargs, oracle_kwargs=oracle_kwargs)
                 
# build the new task (and train a model)         
task = design_bench.make("Quadratic-FullyConnected-v0")

def solve_optimization_problem(x0, y0):
    return x0  # solve a model-based optimization problem

# evaluate the performance of the solution x_star
x_star = solve_optimization_problem(task.x, task.y)
y_star = task.predict(x_star)
```

</details>

## Citation

Thanks for using our benchmark, and please cite our paper!

```
@misc{
    trabucco2021designbench,
    title={Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization},
    author={Brandon Trabucco and Aviral Kumar and Xinyang Geng and Sergey Levine},
    year={2021},
    url={https://openreview.net/forum?id=cQzf26aA3vM}
}
```
