# Design Benchmarks for Model-Based Optimization

This repository contains several design benchmarks for model-based optimization. Our hope is that a common interface and stable nomenclature will encourage future research and comparability in model-based design.

## Available Tasks

We provide the following list of tasks and a corresponding build snippet.

* GFP Protein Fluorescence: `design_bench.make('GFP-v0')`
* Superconductor Critical Temperature: `design_bench.make('Superconductor-v0')`
* Hopper Controller: `design_bench.make('HopperController-v0')`
* 1-Dimension GP Function: `design_bench.make('GP1D-v0')`
* 2-Dimension GP Function: `design_bench.make('GP2D-v0')`

## Setup

You can install our benchmarks with the following command.

```bash
pip install -e git+git://github.com/brandontrabucco/design-bench.git#egg=design_bench
```

## Usage

Every task inherits from the `design_bench.task.Task` class. This class provides access to attributes `task.x` and `task.y` that correspond to designs and labels as numpy arrays, respectively. In addition, every task implements a `task.score(x)` function that provides an (approximate) oracle predictor for `task.y`.

```python
import design_bench
task = design_bench.make('HopperController-v0')
designs = task.x[:10]
real_score = task.y[:10]
oracle_score = task.score(designs)
```

## Contributing

To register new tasks with `design_bench` you need to call the `register` function. For example, suppose we have a custom module named `hello.world.task` that contains a custom task class `HelloWorldTask`.

```python
import design_bench
design_bench.register(
    'HelloWorld-v0',
    'hello.world.task:HelloWorldTask',
    kwargs=dict(hello='world'))
```
