from design_bench.datasets.discrete import TFBind10Dataset
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn import GaussianProcessOracle
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.oracles.tensorflow import FullyConnectedOracle
from design_bench.oracles.tensorflow import LSTMOracle
from design_bench.oracles.tensorflow import ResNetOracle


if __name__ == "__main__":

    model = LSTMOracle(TFBind10Dataset(),
                       disk_target="test3",
                       is_absolute=True,
                       learning_rate=0.001,
                       batch_size=2048)
    print(model.model["rank_correlation"])
