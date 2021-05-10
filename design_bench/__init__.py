from design_bench.registration import registry, register, make, spec
from design_bench.oracles.sklearn.kernels import ProteinKernel


register('GFP-GP-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(noise_std=0.0,
                            max_samples=1000,
                            max_percentile=100,
                            min_percentile=50,
                            kernel=ProteinKernel()))


register('GFP-FullyConnected-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(noise_std=0.0,
                            max_samples=None,
                            max_percentile=100,
                            min_percentile=50,
                            embedding_size=64,
                            hidden_size=512,
                            activation='relu',
                            num_layers=2,
                            epochs=50,
                            shuffle_buffer=5000,
                            learning_rate=0.001,
                            split_kwargs=dict(
                                val_fraction=0.1,
                                subset=None,
                                shard_size=5000,
                                to_disk=True,
                                disk_target="gfp/gfp",
                                is_absolute=False)))


register('GFP-LSTM-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.tensorflow:LSTMOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),

         # keyword arguments for training LSTM oracle
         oracle_kwargs=dict(noise_std=0.0,
                            max_samples=None,
                            max_percentile=100,
                            min_percentile=0,
                            hidden_size=64,
                            num_layers=2,
                            epochs=50,
                            shuffle_buffer=5000,
                            learning_rate=0.01,
                            split_kwargs=dict(
                                val_fraction=0.1,
                                subset=None,
                                shard_size=5000,
                                to_disk=True,
                                disk_target="gfp/gfp",
                                is_absolute=False)))


register('GFP-ResNet-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.tensorflow:ResNetOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),

         # keyword arguments for training ResNet oracle
         oracle_kwargs=dict(noise_std=0.0,
                            max_samples=None,
                            max_percentile=100,
                            min_percentile=50,
                            hidden_size=512,
                            activation='relu',
                            kernel_size=3,
                            num_blocks=4,
                            epochs=25,
                            shuffle_buffer=5000,
                            learning_rate=0.001,
                            split_kwargs=dict(
                                val_fraction=0.1,
                                subset=None,
                                shard_size=5000,
                                to_disk=True,
                                disk_target="gfp/gfp",
                                is_absolute=False)))


register('GFP-Transformer-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.tensorflow:TransformerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),

         # keyword arguments for training Transformer oracle
         oracle_kwargs=dict(noise_std=0.0,
                            max_samples=None,
                            max_percentile=100,
                            min_percentile=50,
                            hidden_size=512,
                            feed_forward_size=2048,
                            activation='relu',
                            num_heads=8,
                            num_blocks=4,
                            epochs=25,
                            shuffle_buffer=5000,
                            learning_rate=0.0001,
                            decay_rate=0.95,
                            split_kwargs=dict(
                                val_fraction=0.1,
                                subset=None,
                                shard_size=5000,
                                to_disk=True,
                                disk_target="gfp/gfp",
                                is_absolute=False)))


register('TFBind8-Exact-v0',
         'design_bench.datasets.discrete:TFBind8Dataset',
         'design_bench.oracles.exact:TFBind8Oracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(max_samples=None,
                             max_percentile=50,
                             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(noise_std=0.0))


register('TFBind10-Exact-v0',
         'design_bench.datasets.discrete:TFBind10Dataset',
         'design_bench.oracles.exact:TFBind10Oracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(max_samples=None,
                             max_percentile=50,
                             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(noise_std=0.0))


register('NASBench-Exact-v0',
         'design_bench.datasets.discrete:NASBenchDataset',
         'design_bench.oracles.exact:NASBenchOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(max_samples=None,
                             max_percentile=30,
                             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(noise_std=0.0))


register('HopperController-Exact-v0',
         'design_bench.datasets.continuous:HopperControllerDataset',
         'design_bench.oracles.exact:HopperControllerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(max_samples=None,
                             max_percentile=100,
                             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(noise_std=0.0))
