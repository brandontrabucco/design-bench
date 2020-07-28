# Design Benchmarks for Model-Based Optimization

This repository contains several benchmarks of design problems for model-based optimization.

In particular, we provide the following family of design problems:

* MuJoCo Hopper Controller Optimization 
* Fluorescent Protein Design
* 1D GP Function Optimization
* 2D GP Function Optimization
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

## Contributing

To register new tasks with `design_bench` you need to call the `register` function. For example, suppose we have a custom module named `hello.world.task` that contains a custom task class `HelloWorldTask`.

```python
import design_bench
design_bench.register(
    'HelloWorld-v0',
    'hello.world.task:HelloWorldTask',
    kwargs={
        'hello': 'world'
    }
)
```
