from design_bench.registration import registry, register, make, spec
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn.kernels import DefaultSequenceKernel
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


register('GFP-GP-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ProteinKernel()),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-RandomForest-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=10,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-FullyConnected-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(embedding_size=64,
                               hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-LSTM-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.tensorflow:LSTMOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for training LSTM oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               num_layers=2,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-ResNet-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.tensorflow:ResNetOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for training ResNet oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               activation='relu',
                               kernel_size=3,
                               num_blocks=4,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-Transformer-v0',
         'design_bench.datasets.discrete:GFPDataset',
         'design_bench.oracles.tensorflow:TransformerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for training Transformer oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             internal_batch_size=32,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               feed_forward_size=256,
                               activation='relu',
                               num_heads=2,
                               num_blocks=4,
                               epochs=20,
                               shuffle_buffer=60000,
                               learning_rate=0.0001,
                               dropout_rate=0.1),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('TFBind8-Exact-v0',
         'design_bench.datasets.discrete:TFBind8Dataset',
         'design_bench.oracles.exact:TFBind8Oracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('TFBind8-GP-v0',
         'design_bench.datasets.discrete:TFBind8Dataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=DefaultSequenceKernel(size=4)),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-RandomForest-v0',
         'design_bench.datasets.discrete:TFBind8Dataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=10,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-FullyConnected-v0',
         'design_bench.datasets.discrete:TFBind8Dataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(embedding_size=64,
                               hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-LSTM-v0',
         'design_bench.datasets.discrete:TFBind8Dataset',
         'design_bench.oracles.tensorflow:LSTMOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for training LSTM oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               num_layers=2,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.01,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-ResNet-v0',
         'design_bench.datasets.discrete:TFBind8Dataset',
         'design_bench.oracles.tensorflow:ResNetOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for training ResNet oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               activation='relu',
                               kernel_size=3,
                               num_blocks=4,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-Transformer-v0',
         'design_bench.datasets.discrete:TFBind8Dataset',
         'design_bench.oracles.tensorflow:TransformerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for training Transformer oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             internal_batch_size=32,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               feed_forward_size=256,
                               activation='relu',
                               num_heads=2,
                               num_blocks=4,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.0001,
                               dropout_rate=0.1),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind10-Exact-v0',
         'design_bench.datasets.discrete:TFBind10Dataset',
         'design_bench.oracles.exact:TFBind10Oracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor='pho4'),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('NASBench-Exact-v0',
         'design_bench.datasets.discrete:NASBenchDataset',
         'design_bench.oracles.exact:NASBenchOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=30,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('UTR-GP-v0',
         'design_bench.datasets.discrete:UTRDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=DefaultSequenceKernel(size=4)),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="utr/split",
                               is_absolute=False)))


register('UTR-RandomForest-v0',
         'design_bench.datasets.discrete:UTRDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=10,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="utr/split",
                               is_absolute=False)))


register('UTR-FullyConnected-v0',
         'design_bench.datasets.discrete:UTRDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(embedding_size=64,
                               hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="utr/split",
                               is_absolute=False)))


register('UTR-LSTM-v0',
         'design_bench.datasets.discrete:UTRDataset',
         'design_bench.oracles.tensorflow:LSTMOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for training LSTM oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               num_layers=2,
                               epochs=20,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="utr/split",
                               is_absolute=False)))


register('UTR-ResNet-v0',
         'design_bench.datasets.discrete:UTRDataset',
         'design_bench.oracles.tensorflow:ResNetOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for training ResNet oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               activation='relu',
                               kernel_size=3,
                               num_blocks=4,
                               epochs=20,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="utr/split",
                               is_absolute=False)))


register('UTR-Transformer-v0',
         'design_bench.datasets.discrete:UTRDataset',
         'design_bench.oracles.tensorflow:TransformerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for training Transformer oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             internal_batch_size=32,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               feed_forward_size=256,
                               activation='relu',
                               num_heads=2,
                               num_blocks=4,
                               epochs=20,
                               shuffle_buffer=5000,
                               learning_rate=0.0001,
                               dropout_rate=0.1),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="utr/split",
                               is_absolute=False)))


register('ChEMBL-GP-v0',
         'design_bench.datasets.discrete:ChEMBLDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=40,
             min_percentile=0,
             assay_chembl_id="CHEMBL1964047",
             standard_type="GI50"),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=53,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=DefaultSequenceKernel(size=591),
                               alpha=0.01),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="chembl-GI50-CHEMBL1964047/split",
                               is_absolute=False)))


register('ChEMBL-RandomForest-v0',
         'design_bench.datasets.discrete:ChEMBLDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=40,
             min_percentile=0,
             assay_chembl_id="CHEMBL1964047",
             standard_type="GI50"),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=53,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=10,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="chembl-GI50-CHEMBL1964047/split",
                               is_absolute=False)))


register('ChEMBL-FullyConnected-v0',
         'design_bench.datasets.discrete:ChEMBLDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=40,
             min_percentile=0,
             assay_chembl_id="CHEMBL1964047",
             standard_type="GI50"),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=53,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(embedding_size=64,
                               hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="chembl-GI50-CHEMBL1964047/split",
                               is_absolute=False)))


register('ChEMBL-LSTM-v0',
         'design_bench.datasets.discrete:ChEMBLDataset',
         'design_bench.oracles.tensorflow:LSTMOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=40,
             min_percentile=0,
             assay_chembl_id="CHEMBL1964047",
             standard_type="GI50"),

         # keyword arguments for training LSTM oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=53,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               num_layers=2,
                               epochs=20,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="chembl-GI50-CHEMBL1964047/split",
                               is_absolute=False)))


register('ChEMBL-ResNet-v0',
         'design_bench.datasets.discrete:ChEMBLDataset',
         'design_bench.oracles.tensorflow:ResNetOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=40,
             min_percentile=0,
             assay_chembl_id="CHEMBL1964047",
             standard_type="GI50"),

         # keyword arguments for training ResNet oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=53,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               activation='relu',
                               kernel_size=3,
                               num_blocks=4,
                               epochs=20,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="chembl-GI50-CHEMBL1964047/split",
                               is_absolute=False)))


register('ChEMBL-Transformer-v0',
         'design_bench.datasets.discrete:ChEMBLDataset',
         'design_bench.oracles.tensorflow:TransformerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=40,
             min_percentile=0,
             assay_chembl_id="CHEMBL1964047",
             standard_type="GI50"),

         # keyword arguments for training Transformer oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             internal_batch_size=32,
             max_samples=None,
             max_percentile=53,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=128,
                               feed_forward_size=512,
                               activation='relu',
                               num_heads=4,
                               num_blocks=4,
                               epochs=20,
                               shuffle_buffer=20000,
                               learning_rate=0.0001,
                               dropout_rate=0.2),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="chembl-GI50-CHEMBL1964047/split",
                               is_absolute=False)))


register('HopperController-Exact-v0',
         'design_bench.datasets.continuous:HopperControllerDataset',
         'design_bench.oracles.exact:HopperControllerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=100,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('HopperController-GP-v0',
         'design_bench.datasets.continuous:HopperControllerDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=100,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ConstantKernel(
                 constant_value=1.0, constant_value_bounds=(0.0, 10.0)) *
                 RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) +
                 RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="hopper_controller/split",
                               is_absolute=False)))


register('HopperController-RandomForest-v0',
         'design_bench.datasets.continuous:HopperControllerDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=100,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=10,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="hopper_controller/split",
                               is_absolute=False)))


register('HopperController-FullyConnected-v0',
         'design_bench.datasets.continuous:HopperControllerDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=100,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=20,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="hopper_controller/split",
                               is_absolute=False)))


register('Superconductor-GP-v0',
         'design_bench.datasets.continuous:SuperconductorDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=80,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ConstantKernel(
                 constant_value=1.0, constant_value_bounds=(0.0, 10.0)) *
                 RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) +
                 RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="superconductor/split",
                               is_absolute=False)))


register('Superconductor-RandomForest-v0',
         'design_bench.datasets.continuous:SuperconductorDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=80,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=10,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="superconductor/split",
                               is_absolute=False)))


register('Superconductor-FullyConnected-v0',
         'design_bench.datasets.continuous:SuperconductorDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=80,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="superconductor/split",
                               is_absolute=False)))


register('AntMorphology-Exact-v0',
         'design_bench.datasets.continuous:AntMorphologyDataset',
         'design_bench.oracles.exact:AntMorphologyOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=20,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('AntMorphology-GP-v0',
         'design_bench.datasets.continuous:AntMorphologyDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=20,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ConstantKernel(
                 constant_value=1.0, constant_value_bounds=(0.0, 10.0)) *
                 RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) +
                 RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="ant_morphology/split",
                               is_absolute=False)))


register('AntMorphology-RandomForest-v0',
         'design_bench.datasets.continuous:AntMorphologyDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=20,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=10,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="ant_morphology/split",
                               is_absolute=False)))


register('AntMorphology-FullyConnected-v0',
         'design_bench.datasets.continuous:AntMorphologyDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=20,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="ant_morphology/split",
                               is_absolute=False)))


register('DKittyMorphology-Exact-v0',
         'design_bench.datasets.continuous:DKittyMorphologyDataset',
         'design_bench.oracles.exact:DKittyMorphologyOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=20,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('DKittyMorphology-GP-v0',
         'design_bench.datasets.continuous:DKittyMorphologyDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=20,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ConstantKernel(
                 constant_value=1.0, constant_value_bounds=(0.0, 10.0)) *
                 RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) +
                 RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="dkitty_morphology/split",
                               is_absolute=False)))


register('DKittyMorphology-RandomForest-v0',
         'design_bench.datasets.continuous:DKittyMorphologyDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=20,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=10,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="dkitty_morphology/split",
                               is_absolute=False)))


register('DKittyMorphology-FullyConnected-v0',
         'design_bench.datasets.continuous:DKittyMorphologyDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             max_percentile=20,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="dkitty_morphology/split",
                               is_absolute=False)))
