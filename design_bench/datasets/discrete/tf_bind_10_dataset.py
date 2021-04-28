from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource


TF_BIND_10_FILES = ["tf_bind_10/tf_bind_10-cbf1-x-11.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-44.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-53.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-20.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-50.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-26.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-83.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-64.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-40.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-55.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-41.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-57.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-77.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-21.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-36.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-61.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-48.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-60.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-8.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-3.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-24.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-6.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-81.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-39.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-4.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-17.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-37.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-54.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-47.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-69.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-7.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-30.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-5.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-63.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-28.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-70.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-29.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-37.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-43.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-46.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-38.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-7.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-34.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-26.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-49.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-22.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-62.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-27.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-57.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-51.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-55.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-58.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-29.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-36.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-1.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-1.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-60.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-74.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-32.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-16.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-67.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-75.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-2.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-61.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-40.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-0.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-33.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-49.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-79.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-13.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-9.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-33.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-42.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-12.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-44.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-35.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-0.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-71.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-72.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-15.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-76.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-23.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-13.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-46.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-38.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-14.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-35.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-11.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-23.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-27.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-8.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-45.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-12.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-24.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-21.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-68.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-52.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-16.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-59.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-53.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-54.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-20.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-19.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-66.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-25.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-48.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-56.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-47.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-80.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-9.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-43.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-30.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-52.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-19.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-25.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-18.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-28.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-6.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-31.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-2.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-62.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-45.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-31.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-39.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-10.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-22.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-14.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-82.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-4.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-3.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-51.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-5.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-42.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-56.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-17.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-34.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-15.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-78.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-65.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-58.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-59.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-73.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-18.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-41.npy",
                    "tf_bind_10/tf_bind_10-cbf1-x-32.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-50.npy",
                    "tf_bind_10/tf_bind_10-pho4-x-10.npy"]


class TfBind10Dataset(DiscreteDataset):
    """A polypeptide synthesis dataset that defines a common set of functions
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

    --- for discrete tasks only

    to_logits(np.ndarray) > np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution

    to_integers(np.ndarray) > np.ndarray:
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

    @staticmethod
    def register_x_shards(transcription_factor='pho4'):
        """Registers a remote file for download that contains design values
        in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        transcription_factor: str
            a string argument that specifies which transcription factor to
            select for model-based optimization, where the goal is to find
            a length 10 polypeptide with maximum binding affinity

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [DiskResource(
            file, is_absolute=False,
            download_target=f"https://design-bench."
                            f"s3-us-west-1.amazonaws.com/{file}",
            download_method="direct") for file in TF_BIND_10_FILES
            if transcription_factor in file]

    @staticmethod
    def register_y_shards(transcription_factor='pho4'):
        """Registers a remote file for download that contains prediction
        values in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        transcription_factor: str
            a string argument that specifies which transcription factor to
            select for model-based optimization, where the goal is to find
            a length 10 polypeptide with maximum binding affinity

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [DiskResource(
            file.replace("-x-", "-y-"), is_absolute=False,
            download_target=f"https://design-bench."
                            f"s3-us-west-1.amazonaws.com/"
                            f"{file.replace('-x-', '-y-')}",
            download_method="direct") for file in TF_BIND_10_FILES
            if transcription_factor in file]

    def __init__(self, soft_interpolation=0.6,
                 transcription_factor='pho4', **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution

        Arguments:

        soft_interpolation: float
            a floating point hyper parameter used when converting design values
            from integers to a floating point representation as logits, which
            interpolates between a uniform and dirac distribution
            1.0 = dirac, 0.0 -> uniform
        transcription_factor: str
            a string argument that specifies which transcription factor to
            select for model-based optimization, where the goal is to find
            a length 10 polypeptide with maximum binding affinity
        **kwargs: dict
            additional keyword arguments which are used to parameterize the
            data set generation process, including which shard files are used
            if multiple sets of data set shard files can be loaded

        """

        # initialize the dataset using the method in the base class
        super(TfBind10Dataset, self).__init__(
            self.register_x_shards(transcription_factor=transcription_factor),
            self.register_y_shards(transcription_factor=transcription_factor),
            is_logits=False, num_classes=4,
            soft_interpolation=soft_interpolation, **kwargs)
