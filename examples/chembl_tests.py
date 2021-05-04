from design_bench.datasets.discrete.gfp_dataset import GFPDataset
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn import GaussianProcessOracle
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.oracles.tensorflow import FullyConnectedOracle
from design_bench.oracles.tensorflow import LSTMOracle
from design_bench.oracles.tensorflow import ResNetOracle
from scipy import stats


if __name__ == "__main__":

    resnet = ResNetOracle(GFPDataset())
    print(resnet.model["rank_correlation"])
