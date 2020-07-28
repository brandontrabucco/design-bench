# Design Benchmarks for Model-Based Optimization

This repository contains several benchmarks of design problems for model-based optimization.

In particular, we provide the following family of design problems:

* MuJoCo Hopper Policy Optimization 
* Fluorescent Protein Design
* ROBEL D'Kitty Morphology Design
* MuJoCo Ant Morphology Design 
* MuJoCo Dog Morphology Design 

## Setup

You can install our benchmarks with the following command.

```bash
pip install git+git://github.com/brandontrabucco/design-bench.git
```

## Usage

You can instantiate a design problem using the `make` function. Note that the first time you import `design_bench` many data files will be downloaded, so the first import may be slow.

```python
import design_bench
task = design_bench.make('HopperController-v0')
```
