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


register('TFBind8-Exact-v0',
         'design_bench.datasets.discrete:TFBind8Dataset',
         'design_bench.oracles.exact:TFBind8Oracle',
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),
         oracle_kwargs=dict(noise_std=0.0))


register('TFBind10-Exact-v0',
         'design_bench.datasets.discrete:TFBind10Dataset',
         'design_bench.oracles.exact:TFBind10Oracle',
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),
         oracle_kwargs=dict(noise_std=0.0))


register('NASBench-Exact-v0',
         'design_bench.datasets.discrete:NASBenchDataset',
         'design_bench.oracles.exact:NASBenchOracle',
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),
         oracle_kwargs=dict(noise_std=0.0))


register('HopperController-Exact-v0',
         'design_bench.datasets.continuous:HopperControllerDataset',
         'design_bench.oracles.exact:HopperControllerOracle',
         dataset_kwargs=dict(max_samples=5000,
                             max_percentile=60,
                             min_percentile=50),
         oracle_kwargs=dict(noise_std=0.0))
