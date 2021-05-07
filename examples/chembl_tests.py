from design_bench.datasets.discrete import ChEMBLDataset
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn import GaussianProcessOracle
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.oracles.tensorflow import FullyConnectedOracle
from design_bench.oracles.tensorflow import LSTMOracle
from design_bench.oracles.tensorflow import ResNetOracle
from design_bench.oracles.tensorflow import TransformerOracle
from scipy import stats


if __name__ == "__main__":

    d = ChEMBLDataset()
    d.subsample(max_percentile=53)
    resnet = TransformerOracle(d)
    print(resnet.model["rank_correlation"])
