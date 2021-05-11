from design_bench.datasets.discrete import GFPDataset
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn import GaussianProcessOracle
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.oracles.tensorflow import FullyConnectedOracle
from design_bench.oracles.tensorflow import LSTMOracle
from design_bench.oracles.tensorflow import ResNetOracle
from design_bench.oracles.tensorflow import TransformerOracle
from scipy import stats
import numpy as np
import pandas as pd


if __name__ == "__main__":

    dataset = GFPDataset()
    oracle = TransformerOracle(dataset, fit=True,
                               learning_rate=0.0001, epochs=50)
