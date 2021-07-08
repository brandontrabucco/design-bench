from design_bench.datasets.dataset_builder import DatasetBuilder
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.oracles.oracle_builder import OracleBuilder
from design_bench.oracles.approximate_oracle import ApproximateOracle
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
    """A container class for model-based optimization problems where a
    dataset is paired with a ground truth score function such as a
    neural network learned with gradient descent, where the goal
    is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    is_discrete: bool
        An attribute that specifies whether the task dataset is discrete or
        continuous determined by whether the dataset instance task.dataset
        inherits from DiscreteDataset or ContinuousDataset

    dataset_name: str
        An attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench
    oracle_name: str
        Attribute that specifies the name of a machine learning model used
        as the ground truth score function for a model-based optimization
        problem; for example, "random_forest"
    x_name: str
        An attribute that specifies the name of designs in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper
    y_name: str
        An attribute that specifies the name of predictions in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

    x: np.ndarray
        the design values 'x' for a model-based optimization problem
        represented as a numpy array of arbitrary type
    input_shape: Tuple[int]
        the shape of a single design values 'x', represented as a list of
        integers similar to calling np.ndarray.shape
    input_size: int
        the total number of components in the design values 'x', represented
        as a single integer, the product of its shape entries
    input_dtype: np.dtype
        the data type of the design values 'x', which is typically either
        floating point or integer (np.float32 or np.int32)

    y: np.ndarray
        the prediction values 'y' for a model-based optimization problem
        represented by a scalar floating point value per 'x'
    output_shape: Tuple[int]
        the shape of a single prediction value 'y', represented as a list of
        integers similar to calling np.ndarray.shape
    output_size: int
        the total number of components in the prediction values 'y',
        represented as a single integer, the product of its shape entries
    output_dtype: np.dtype
        the data type of the prediction values 'y', which is typically a
        type of floating point (np.float32 or np.float16)

    dataset_size: int
        the total number of paired design values 'x' and prediction values
        'y' in the dataset, represented as a single integer
    dataset_distribution: Callable[np.ndarray, np.ndarray]
        the target distribution of the model-based optimization dataset
        marginal p(y) used for controlling the sampling distribution
    dataset_max_percentile: float
        the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_percentile: float
        the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class
    dataset_max_output: float
        the specific cutoff threshold for prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_output: float
        the specific cutoff threshold for prediction values 'y' below
        which are hidden from access by members outside the class

    is_normalized_x: bool
        a boolean indicator that specifies whether the design values in
        the dataset are being normalized
    is_normalized_y: bool
        a boolean indicator that specifies whether the prediction values
        in the dataset are being normalized

    --- for discrete tasks only

    is_logits: bool (only supported for a DiscreteDataset)
        a value that indicates whether the design values contained in the
        model-based optimization dataset have already been converted to
        logits and need not be converted again
    num_classes: int
        an integer representing the number of classes in the distribution
        that the integer data points are sampled from which cannot be None
        and must also be greater than 1

    Public Methods:

    predict(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    dataset_to_oracle_x(np.ndarray) -> np.ndarray
        Helper function for converting from designs contained in the
        dataset format into a format the oracle is expecting to process,
        such as from integers to logits of a categorical distribution
    dataset_to_oracle_y(np.ndarray) -> np.ndarray
        Helper function for converting from predictions contained in the
        dataset format into a format the oracle is expecting to process,
        such as from normalized to denormalized predictions
    oracle_to_dataset_x(np.ndarray) -> np.ndarray
        Helper function for converting from designs in the format of the
        oracle into the design format the dataset contains, such as
        from categorical logits to integers
    oracle_to_dataset_y(np.ndarray) -> np.ndarray
        Helper function for converting from predictions in the
        format of the oracle into a format the dataset contains,
        such as from normalized to denormalized predictions

    iterate_batches(batch_size: int, return_x: bool,
                    return_y: bool, drop_remainder: bool)
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model
    iterate_samples(return_x: bool, return_y: bool):
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

    normalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and undoes standardization so that they have their
        original empirical mean and variance
    normalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and undoes standardization so that they have their
        original empirical mean and variance

    map_normalize_x():
        a destructive function that standardizes the design values 'x'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_x():
        a destructive function that undoes standardization of the
        design values 'x' in the class dataset in-place which are expected
        to have zero  empirical mean and unit variance
    map_normalize_y():
        a destructive function that standardizes the prediction values 'y'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_y():
        a destructive function that undoes standardization of the
        prediction values 'y' in the class dataset in-place which are
        expected to have zero empirical mean and unit variance

    --- for discrete tasks only

    to_logits(np.ndarray) -> np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution
    to_integers(np.ndarray) -> np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of floating point logits as input and converts them to integer
        representing the max of the distribution

    map_to_logits():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits
    map_to_integers():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

    """

    def __init__(self, dataset: Union[DatasetBuilder, type, str],
                 oracle: Union[OracleBuilder, type, str],
                 dataset_kwargs=None, oracle_kwargs=None, relabel=False):
        """Initialize a model-based optimization problem using a static task
        dataset and a ground truth score function that is either an exact
        simulator, or an approximate model such as a neural network

        Arguments:

        dataset: Union[DatasetBuilder, type, str]
            a static dataset in a model-based optimization problem that
            exposes a set of designs 'x' and predictions 'y'
        oracle: Union[OracleBuilder, type, str]
            a ground truth score function in a model-based optimization
            problem that is either an exact simulator or an approximate model
        dataset_kwargs: dict
            additional keyword arguments that are provided to the dataset
            class when it is initialized for the first time
        oracle_kwargs: dict
            additional keyword arguments that are provided to the oracle
            class when it is initialized for the first time
        relabel: bool
            a boolean indicator that specifies whether the dataset prediction
            values should be relabeled with the predictions of the oracle

        """

        # use additional_kwargs to override self.kwargs
        kwargs = dataset_kwargs if dataset_kwargs else dict()

        # if self.entry_point is a function call it
        if callable(dataset):
            dataset = dataset(**kwargs)

        # if self.entry_point is a string import it first
        elif isinstance(dataset, str):
            dataset = import_name(dataset)(**kwargs)

        # return if the dataset could not be loaded
        elif not isinstance(dataset, DatasetBuilder):
            raise ValueError("dataset could not be loaded")

        # expose the built dataset
        self.dataset = dataset

        # use additional_kwargs to override self.kwargs
        kwargs = oracle_kwargs if oracle_kwargs else dict()

        # if self.entry_point is a function call it
        if callable(oracle):
            oracle = oracle(dataset, **kwargs)

        # if self.entry_point is a string import it first
        elif isinstance(oracle, str):
            oracle = import_name(oracle)(dataset, **kwargs)

        # return if the oracle could not be loaded
        elif not isinstance(oracle, OracleBuilder):
            raise ValueError("oracle could not be loaded")

        # expose the built oracle
        self.oracle = oracle

        # only relabel when an approximate model is used
        relabel = relabel and isinstance(oracle, ApproximateOracle)

        # attempt to download the appropriate shards
        new_shards = []
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

            # assign the y shards to the downloaded files and re sample
            # the dataset if sub sampling is being used
            dataset.y_shards = new_shards
            dataset.subsample(max_samples=dataset.dataset_size,
                              distribution=dataset.dataset_distribution,
                              max_percentile=dataset.dataset_max_percentile,
                              min_percentile=dataset.dataset_min_percentile)

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
    def is_discrete(self):
        """Attribute that specifies whether the task dataset is discrete or
        continuous determined by whether the dataset instance task.dataset
        inherits from DiscreteDataset or ContinuousDataset

        """

        return isinstance(self.dataset, DiscreteDataset)

    @property
    def oracle_name(self):
        """Attribute that specifies the name of a machine learning model used
        as the ground truth score function for a model-based optimization
        problem; for example, "random_forest"

        """

        return self.oracle.name

    @property
    def dataset_name(self):
        """An attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench

        """

        return self.dataset.name

    @property
    def x_name(self):
        """An attribute that specifies the name of designs in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

        """

        return self.dataset.x_name

    @property
    def y_name(self):
        """An attribute that specifies the name of predictions in a
        model-based optimization dataset, which might be used when labelling
        plots in a visualization of performance in a research paper

        """

        return self.dataset.y_name

    @property
    def dataset_size(self):
        """the total number of paired design values 'x' and prediction values
        'y' in the dataset, represented as a single integer

        """

        return self.dataset.dataset_size

    @property
    def dataset_distribution(self):
        """the target distribution of the model-based optimization dataset
        marginal p(y) used for controlling the sampling distribution

        """

        return self.dataset.dataset_distribution

    @property
    def dataset_max_percentile(self):
        """the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class

        """

        return self.dataset.dataset_max_percentile

    @property
    def dataset_min_percentile(self):
        """the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class

        """

        return self.dataset.dataset_min_percentile

    @property
    def dataset_max_output(self):
        """the specific cutoff threshold for prediction values 'y' above
        which are hidden from access by members outside the class

        """

        return self.dataset.dataset_max_output

    @property
    def dataset_min_output(self):
        """the specific cutoff threshold for prediction values 'y' below
        which are hidden from access by members outside the class

        """

        return self.dataset.dataset_min_output

    @property
    def input_shape(self):
        """the shape of a single design values 'x', represented as a list of
        integers similar to calling np.ndarray.shape

        """

        return self.dataset.input_shape

    @property
    def input_size(self):
        """the total number of components in the design values 'x',
        represented as a single integer, the product of its shape entries

        """

        return self.dataset.input_size

    @property
    def input_dtype(self):
        """the data type of the design values 'x', which is typically either
        floating point or integer (np.float32 or np.int32)

        """

        return self.dataset.input_dtype

    @property
    def output_shape(self):
        """the shape of a single prediction value 'y', represented as a list
        of integers similar to calling np.ndarray.shape

        """

        return self.dataset.output_shape

    @property
    def output_size(self):
        """the total number of components in the prediction values 'y',
        represented as a single integer, the product of its shape entries

        """

        return self.dataset.output_size

    @property
    def output_dtype(self):
        """the data type of the prediction values 'y', which is typically a
        type of floating point (np.float32 or np.float16)

        """

        return self.dataset.output_dtype

    @property
    def x(self):
        """the design values 'x' for a model-based optimization problem
        represented as a numpy array of arbitrary type

        """

        return self.dataset.x

    @property
    def y(self):
        """the prediction values 'y' for a model-based optimization problem
        represented by a scalar floating point value per 'x'

        """

        return self.dataset.y

    @property
    def is_normalized_x(self):
        """a boolean indicator that specifies whether the design values in
        the dataset are being normalized

        """

        return self.dataset.is_normalized_x

    @property
    def is_normalized_y(self):
        """a boolean indicator that specifies whether the prediction values
        in the dataset are being normalized

        """

        return self.dataset.is_normalized_y

    @property
    def is_logits(self):
        """a value that indicates whether the design values contained in the
        model-based optimization dataset have already been converted to
        logits and need not be converted again

        """

        if not hasattr(self.dataset, "is_logits"):
            raise ValueError("only supported on discrete datasets")
        return self.dataset.is_logits

    @property
    def num_classes(self):
        """an integer representing the number of classes in the distribution
        that the integer data points are sampled from which cannot be None
        and must also be greater than 1

        """

        if not hasattr(self.dataset, "num_classes"):
            raise ValueError("only supported on discrete datasets")
        return self.dataset.num_classes

    def iterate_batches(self, batch_size, return_x=True,
                        return_y=True, drop_remainder=False):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Arguments:

        batch_size: int
            a positive integer that specifies the batch size of samples
            taken from a model-based optimization data set; batches
            with batch_size elements are yielded
        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True
        drop_remainder: bool
            a boolean indicator representing whether the last batch
            should be dropped in the case it has fewer than batch_size
            elements; the default behavior is not to drop the smaller batch.

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        return iter(self.dataset.iterate_batches(
            batch_size, return_x=return_x,
            return_y=return_y, drop_remainder=drop_remainder))

    def iterate_samples(self, return_x=True, return_y=True):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Arguments:

        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        return iter(self.dataset.iterate_samples(return_x=return_x,
                                                 return_y=return_y))

    def __iter__(self):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        return iter(self.dataset)

    def map_normalize_x(self):
        """a function that standardizes the design values 'x' to have zero
        empirical mean and unit empirical variance in the dataset

        """

        self.dataset.map_normalize_x()

    def map_normalize_y(self):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance in the dataset

        """

        self.dataset.map_normalize_y()

    def map_denormalize_x(self):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance in the dataset

        """

        self.dataset.map_denormalize_x()

    def map_denormalize_y(self):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance in the dataset

        """

        self.dataset.map_denormalize_y()

    def map_to_integers(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

        """

        if not hasattr(self.dataset, "map_to_integers"):
            raise ValueError("only supported on discrete datasets")
        self.dataset.map_to_integers()

    def map_to_logits(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits

        """

        if not hasattr(self.dataset, "map_to_logits"):
            raise ValueError("only supported on discrete datasets")
        self.dataset.map_to_logits()

    def normalize_x(self, x):
        """a function that standardizes the design values 'x' to have
        zero empirical mean and unit empirical variance

        Arguments:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            shall be normalized according to dataset statistics

        Returns:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            has been normalized using dataset statistics

        """

        return self.dataset.normalize_x(x)

    def normalize_y(self, y):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance

        Arguments:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            shall be normalized according to dataset statistics

        Returns:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            has been normalized using dataset statistics

        """

        return self.dataset.normalize_y(y)

    def denormalize_x(self, x):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance

        Arguments:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            shall be denormalized according to dataset statistics

        Returns:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            has been denormalized using dataset statistics

        """

        return self.dataset.denormalize_x(x)

    def denormalize_y(self, y):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance

        Arguments:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            shall be denormalized according to dataset statistics

        Returns:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            has been denormalized using dataset statistics

        """

        return self.dataset.denormalize_y(y)

    def to_integers(self, x):
        """A helper function that accepts design values represented as a numpy
        array of floating point logits as input and converts them to integer
        representing the max of the distribution

        Arguments:

        x: np.ndarray
            a numpy array containing design values represented as floating
            point numbers which have be converted from integer samples of
            a certain probability distribution

        Returns:

        x: np.ndarray
            a numpy array containing design values represented as integers
            which have been converted from a floating point
            representation of a certain probability distribution

        """

        if not hasattr(self.dataset, "map_to_integers"):
            raise ValueError("only supported on discrete datasets")
        return self.dataset.to_integers(x)

    def to_logits(self, x):
        """A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution

        Arguments:

        x: np.ndarray
            a numpy array containing design values represented as integers
            which are going to be converted into a floating point
            representation of a certain probability distribution

        Returns:

        x: np.ndarray
            a numpy array containing design values represented as floating
            point numbers which have be converted from integer samples of
            a certain probability distribution

        """

        if not hasattr(self.dataset, "map_to_logits"):
            raise ValueError("only supported on discrete datasets")
        return self.dataset.to_logits(x)

    def predict(self, x_batch, **kwargs):
        """a function that accepts a batch of design values 'x' as input and
        for each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        return self.oracle.predict(x_batch, **kwargs)

    def oracle_to_dataset_x(self, x_batch):
        """Helper function for converting from designs in the format of the
        oracle into the design format the dataset contains, such as
        from categorical logits to integers

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from
            the format of designs contained in the dataset to the
            format expected by the oracle score function

        Returns:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from
            the format of the oracle to the format of designs
            contained in the dataset

        """

        return self.oracle.oracle_to_dataset_x(x_batch)

    def oracle_to_dataset_y(self, y_batch):
        """Helper function for converting from predictions in the
        format of the oracle into a format the dataset contains,
        such as from normalized to denormalized predictions

        Arguments:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of predictions contained in the dataset to the
            format expected by the oracle score function

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of the oracle to the format of predictions
            contained in the dataset

        """

        return self.oracle.oracle_to_dataset_y(y_batch)

    def dataset_to_oracle_x(self, x_batch):
        """Helper function for converting from designs contained in the
        dataset format into a format the oracle is expecting to process,
        such as from integers to logits of a categorical distribution

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from the
            format of designs contained in the dataset to the format
            expected by the oracle score function

        """

        return self.oracle.dataset_to_oracle_x(x_batch)

    def dataset_to_oracle_y(self, y_batch):
        """Helper function for converting from predictions contained in the
        dataset format into a format the oracle is expecting to process,
        such as from normalized to denormalized predictions

        Arguments:

        y_batch: np.ndarray
            a batch of prediction values 'y' that are from the dataset and
            will be processed into a format expected by the oracle score
            function, which is useful when training the oracle

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of predictions contained in the dataset to the
            format expected by the oracle score function

        """

        return self.oracle.dataset_to_oracle_y(y_batch)
