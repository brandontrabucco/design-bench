# Benchmarks for Data-Driven Offline Model-Based Optimization

This repository contains several design benchmarks for model-based optimization. Our hope is that a common interface and stable nomenclature will encourage future research and comparability in model-based design.

## Available Tasks

Current model-based design benchmarks (circa 2020) typically vary from paper-to-paper. For example, tasks employed by biologists differ strongly from those of interest to roboticists. We provide a common interface for tasks that span a wide-range of disciplines, from materials science, to reinforcement learning. We list these tasks below.

* __Biology__: Protein Fluorescence: `design_bench.make('GFP-v0')`
* __Chemistry__: Molecule Activity: `design_bench.make('MoleculeActivity-v0')`
* __Materials Science__: Superconductor Critical Temperature: `design_bench.make('Superconductor-v0')`
* __Robotics__: Hopper Controller: `design_bench.make('HopperController-v0')`
* __Robotics__: Ant Morphology: `design_bench.make('AntMorphology-v0')`
* __Robotics__: DKitty Morphology: `design_bench.make('DKittyMorphology-v0')`

In addition, the following debugging tasks are provided.

* __Debugging__: Quadratic Maximization: `design_bench.make('Quadratic-v0')`

## Setup

You can install our benchmarks with the following command.

```bash
pip install design-bench[all]
```

If you do not have a MuJoCo License you can install the base benchmark.

```bash
pip install design-bench
```

## Usage

Every task inherits from the `design_bench.task.Task` class. This class provides access to attributes `task.x` and `task.y` that correspond to designs and labels as numpy arrays, respectively. In addition, every task implements a `task.score(x)` function that provides an (approximate) oracle predictor for `task.y`.

```python
import design_bench
task = design_bench.make('Superconductor-v0')
x = task.x[:10]
y = task.y[:10]
oracle_y = task.score(x)
```
