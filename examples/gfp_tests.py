from design_bench.datasets.discrete.gfp_dataset import GFPDataset
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn import GaussianProcessOracle
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.oracles.tensorflow import FullyConnectedOracle


if __name__ == "__main__":

    dataset = GFPDataset()
    dataset.map_to_logits()
    dataset.map_normalize_x()
    dataset.map_denormalize_y()
    dataset.subsample(max_samples=5000,
                      min_percentile=50,
                      max_percentile=60)

    gaussian_process = GaussianProcessOracle(dataset, kernel=ProteinKernel())
    random_forest = RandomForestOracle(dataset)
    fully_connected = FullyConnectedOracle(dataset)
