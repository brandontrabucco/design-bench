from design_bench.datasets.dataset_builder import DatasetBuilder
from design_bench.oracles.oracle_builder import OracleBuilder
from design_bench.disk_resource import DiskResource
from typing import Union
import importlib
import os
import re


# used to determine the name of a dataset that is sharded to disk
SHARD_PATTERN = re.compile(r'(.+)-(\w)-(\d+).npy$')


# this is used to import data set classes dynamically
def import_name(name):
    mod_name, attr_name = name.split(":")
    return getattr(importlib.import_module(mod_name), attr_name)


class Task(object):

    def __init__(self, dataset: Union[DatasetBuilder, type, str],
                 oracle: Union[OracleBuilder, type, str],
                 dataset_kwargs=None, oracle_kwargs=None, relabel=True):

        # use additional_kwargs to override self.kwargs
        kwargs = dict()
        if dataset_kwargs is not None:
            kwargs.update(dataset_kwargs)

        # if self.entry_point is a function call it
        if callable(dataset):
            dataset = dataset(**kwargs)

        # if self.entry_point is a string import it first
        elif isinstance(dataset, str):
            dataset = import_name(dataset)(**kwargs)

        # return if the dataset could not be loaded
        else:
            raise ValueError("dataset could not be loaded")

        # use additional_kwargs to override self.kwargs
        kwargs = dict()
        if oracle_kwargs is not None:
            kwargs.update(oracle_kwargs)

        # if self.entry_point is a function call it
        if callable(oracle):
            oracle = oracle(dataset, **kwargs)

        # if self.entry_point is a string import it first
        elif isinstance(oracle, str):
            oracle = import_name(oracle)(dataset, **kwargs)

        # return if the oracle could not be loaded
        else:
            print(oracle)
            raise ValueError("oracle could not be loaded")

        # expose the dataset and oracle model
        self.dataset = dataset
        self.oracle = oracle
        new_shards = []

        # attempt to download the appropriate shards
        for shard in dataset.y_shards:
            if relabel and isinstance(shard, DiskResource):

                # create a name for the new sharded prediction
                m = SHARD_PATTERN.search(shard.disk_target)
                file = f"{m.group(1)}-{oracle.name}-y-{m.group(3)}.npy"
                bare = os.path.join(os.path.basename(os.path.dirname(file)),
                                    os.path.basename(file))

                # create a disk resource for the new shard
                new_shards.append(DiskResource(
                    file, is_absolute=True, download_method="direct",
                    download_target=f"https://design-bench."
                                    f"s3-us-west-1.amazonaws.com/{bare}"))

        # check if every shard was downloaded successfully
        # this naturally handles when the shard is already downloaded
        if relabel and len(new_shards) > 0 and all([
                f.is_downloaded or f.download() for f in new_shards]):
            dataset.y_shards = new_shards

        elif relabel:

            # test if the shards are stored on the disk
            # this means that downloading cached predictions failed
            name = None
            test_shard = dataset.y_shards[0]
            if isinstance(test_shard, DiskResource):

                # create a name for the new sharded prediction
                m = SHARD_PATTERN.search(test_shard.disk_target)
                name = f"{m.group(1)}-{oracle.name}"

            # relabel the dataset using the new oracle model
            dataset.relabel(lambda x, y: oracle.predict(x),
                            to_disk=name is not None,
                            is_absolute=True, disk_target=name)

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
