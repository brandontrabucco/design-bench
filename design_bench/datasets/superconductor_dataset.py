from design_bench.core.continuous_dataset import ContinuousDataset
from design_bench.utils.remote_resource import RemoteResource


class SuperconductorDataset(ContinuousDataset):
    """A superconductivity dataset that defines a common set of functions
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

    x_resources: list of RemoteResource
        a list of RemoteResource that should be downloaded before the
        dataset can be loaded and used for model-based optimization

    y_resources: list of RemoteResource
        a list of RemoteResource that should be downloaded before the
        dataset can be loaded and used for model-based optimization

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

    def __init__(self, **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution

        Arguments:

        **kwargs: dict
            additional keyword arguments passed to the "load_dataset" method,
            which may be data specific and depend on whether the dataset
            contains discrete or continuous data points

        """

        # initialize the dataset using the method in the base class
        super(SuperconductorDataset, self).__init__(**kwargs)

    @staticmethod
    def register_x_resources(**kwargs):
        """Registers a remote file for download that contains design values
        in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        **kwargs: dict
            additional keyword arguments used for building the dataset,
            which may be domain specific and depend on whether the dataset
            contains discrete or continuous data points

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [RemoteResource(f"superconductor/"
                               f"superconductor-x-{i}.npy",
                               is_absolute=False, download_target=None,
                               download_method=None) for i in range(5)]

    @staticmethod
    def register_y_resources(**kwargs):
        """Registers a remote file for download that contains prediction
        values in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        **kwargs: dict
            additional keyword arguments used for building the dataset,
            which may be domain specific and depend on whether the dataset
            contains discrete or continuous data points

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [RemoteResource(f"superconductor/"
                               f"superconductor-y-{i}.npy",
                               is_absolute=False, download_target=None,
                               download_method=None) for i in range(5)]
