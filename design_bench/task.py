from design_bench.datasets.dataset_builder import DatasetBuilder
from design_bench.oracles.oracle_builder import OracleBuilder


class Task(object):

    def __init__(self, dataset: DatasetBuilder, oracle: OracleBuilder):
        self.dataset, self.oracle = dataset, oracle

    @property
    def oracle_name(self):
        return self.oracle.name

    @property
    def dataset_name(self):
        return self.dataset.name

    @property
    def x_name(self):
        return self.dataset.x_name

    @property
    def y_name(self):
        return self.dataset.y_name

    @property
    def dataset_size(self):
        return self.dataset.dataset_size

    @property
    def dataset_max_percentile(self):
        return self.dataset.dataset_max_percentile

    @property
    def dataset_min_percentile(self):
        return self.dataset.dataset_min_percentile

    @property
    def dataset_max_output(self):
        return self.dataset.dataset_max_output

    @property
    def dataset_min_output(self):
        return self.dataset.dataset_min_output

    @property
    def input_shape(self):
        return self.dataset.input_shape

    @property
    def input_size(self):
        return self.dataset.input_size

    @property
    def input_dtype(self):
        return self.dataset.input_dtype

    @property
    def output_shape(self):
        return self.dataset.output_shape

    @property
    def output_size(self):
        return self.dataset.output_size

    @property
    def output_dtype(self):
        return self.dataset.output_dtype

    @property
    def x(self):
        return self.dataset.x

    @property
    def y(self):
        return self.dataset.y

    def iterate_batches(self, batch_size, return_x=True,
                        return_y=True, drop_remainder=False):
        return iter(self.dataset.iterate_batches(
            batch_size, return_x=return_x,
            return_y=return_y, drop_remainder=drop_remainder))

    def iterate_samples(self, return_x=True, return_y=True):
        return iter(self.dataset.iterate_samples(return_x=return_x,
                                                 return_y=return_y))

    def __iter__(self):
        return iter(self.dataset)

    def map_normalize_x(self):
        self.dataset.map_normalize_x()

    def map_normalize_y(self):
        self.dataset.map_normalize_y()

    def map_denormalize_x(self):
        self.dataset.map_denormalize_x()

    def map_denormalize_y(self):
        self.dataset.map_denormalize_y()

    def map_to_integers(self):
        if not hasattr(self.dataset, "map_to_integers"):
            raise ValueError("only supported on discrete datasets")
        self.dataset.map_to_integers()

    def map_to_logits(self):
        if not hasattr(self.dataset, "map_to_logits"):
            raise ValueError("only supported on discrete datasets")
        self.dataset.map_to_logits()

    def normalize_x(self, x):
        return self.dataset.normalize_x(x)

    def normalize_y(self, y):
        return self.dataset.normalize_y(y)

    def denormalize_x(self, x):
        return self.dataset.denormalize_x(x)

    def denormalize_y(self, y):
        return self.dataset.denormalize_y(y)

    def to_integers(self, x):
        if not hasattr(self.dataset, "map_to_integers"):
            raise ValueError("only supported on discrete datasets")
        return self.dataset.to_integers(x)

    def to_logits(self, x):
        if not hasattr(self.dataset, "map_to_logits"):
            raise ValueError("only supported on discrete datasets")
        return self.dataset.to_logits(x)

    def predict(self, x_batch):
        return self.oracle.predict(x_batch)

    def oracle_to_dataset_x(self, x_batch):
        return self.oracle.oracle_to_dataset_x(x_batch)

    def oracle_to_dataset_y(self, y_batch):
        return self.oracle.oracle_to_dataset_y(y_batch)

    def dataset_to_oracle_x(self, x_batch):
        return self.oracle.dataset_to_oracle_x(x_batch)

    def dataset_to_oracle_y(self, y_batch):
        return self.oracle.dataset_to_oracle_y(y_batch)
