from design_bench.datasets.dataset_builder import DatasetBuilder
import numpy as np


class ContinuousDataset(DatasetBuilder):
    """An abstract base class that defines a common set of functions
    and attributes for a model-based optimization dataset, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

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

    dataset_max_percentile: float
        the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class

    dataset_min_percentile: float
        the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class

    Public Methods:

    subsample(max_percentile: float,
              min_percentile: float):
        a function that exposes a subsampled version of a much larger
        model-based optimization dataset containing design values 'x'
        whose prediction values 'y' are skewed

    relabel(relabel_function:
            Callable[[np.ndarray, np.ndarray], np.ndarray]):
        a function that accepts a function that maps from a dataset of
        design values 'x' and prediction values y to a new set of
        prediction values 'y' and relabels the model-based optimization dataset

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

    """

    def rebuild_dataset(self, x_shards, y_shards, **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution of designs and predictions

        Arguments:

        x_shards: Union[         np.ndarray,           RemoteResource,
                        Iterable[np.ndarray], Iterable[RemoteResource]]
            a single shard or a list of shards representing the design values
            in a model-based optimization dataset; shards are loaded lazily
            if RemoteResource otherwise loaded in memory immediately
        y_shards: Union[         np.ndarray,           RemoteResource,
                        Iterable[np.ndarray], Iterable[RemoteResource]]
            a single shard or a list of shards representing prediction values
            in a model-based optimization dataset; shards are loaded lazily
            if RemoteResource otherwise loaded in memory immediately
        **kwargs: dict
            additional keyword arguments used by sub classes that determine
            functionality or apply transformations to a model-based
            optimization dataset such as an internal batch size

        Returns:

        dataset: DatasetBuilder
            an instance of a data set builder subclass containing a copy
            of all statistics associated with this dataset

        """

        # new dataset that shares statistics with this one
        dataset = ContinuousDataset(
            x_shards, y_shards,
            internal_batch_size=kwargs.get(
                "internal_batch_size", self.internal_batch_size))

        # carry over the normalize statistics of the parent
        dataset.is_normalized_x = self.is_normalized_x
        dataset.x_mean = self.x_mean
        dataset.x_standard_dev = self.x_standard_dev

        # carry over the normalize statistics of the parent
        dataset.is_normalized_y = self.is_normalized_y
        dataset.y_mean = self.y_mean
        dataset.y_standard_dev = self.y_standard_dev

        # carry over the sub sampling statistics of the parent
        dataset.dataset_min_percentile = self.dataset_min_percentile
        dataset.dataset_max_percentile = self.dataset_max_percentile
        dataset.dataset_min_output = self.dataset_min_output
        dataset.dataset_max_output = self.dataset_max_output
        dataset.dataset_size = dataset.y.shape[0]

        # return the new dataset
        return dataset
