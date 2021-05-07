from design_bench.registration import registry, register, make, spec
from design_bench.oracles.sklearn.kernels import ProteinKernel


register('GFP-GP-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),
         oracle_kwargs=dict(noise_std=0.0,
                            max_samples=1000,
                            max_percentile=100,
                            min_percentile=50,
                            kernel=ProteinKernel()))
