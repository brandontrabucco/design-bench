from design_bench.datasets.discrete.gfp_dataset import GFPDataset
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn import GaussianProcessOracle
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.oracles.tensorflow import FullyConnectedOracle
from design_bench.oracles.tensorflow import LSTMOracle
from design_bench.oracles.tensorflow import ResNetOracle
from design_bench.oracles.tensorflow import TransformerOracle


if __name__ == "__main__":

    model = TransformerOracle(GFPDataset(),
                              disk_target="test",
                              is_absolute=True,
                              noise_std=0.0,
                              max_samples=None,
                              max_percentile=100,
                              min_percentile=0,
                              hidden_size=64,
                              num_layers=2,
                              epochs=20,
                              shuffle_buffer=5000,
                              learning_rate=0.001,
                              split_kwargs=dict(
                               val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/test",
                               is_absolute=True))

    print(model.model["rank_correlation"])
