from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.remote_resource import RemoteResource


NAS_BENCH_FILES = ["nas_bench/nas_bench-x-40.npy",
                   "nas_bench/nas_bench-x-154.npy",
                   "nas_bench/nas_bench-x-175.npy",
                   "nas_bench/nas_bench-x-250.npy",
                   "nas_bench/nas_bench-x-245.npy",
                   "nas_bench/nas_bench-x-18.npy",
                   "nas_bench/nas_bench-x-12.npy",
                   "nas_bench/nas_bench-x-73.npy",
                   "nas_bench/nas_bench-x-125.npy",
                   "nas_bench/nas_bench-x-75.npy",
                   "nas_bench/nas_bench-x-109.npy",
                   "nas_bench/nas_bench-x-157.npy",
                   "nas_bench/nas_bench-x-209.npy",
                   "nas_bench/nas_bench-x-146.npy",
                   "nas_bench/nas_bench-x-213.npy",
                   "nas_bench/nas_bench-x-3.npy",
                   "nas_bench/nas_bench-x-77.npy",
                   "nas_bench/nas_bench-x-226.npy",
                   "nas_bench/nas_bench-x-78.npy",
                   "nas_bench/nas_bench-x-99.npy",
                   "nas_bench/nas_bench-x-117.npy",
                   "nas_bench/nas_bench-x-24.npy",
                   "nas_bench/nas_bench-x-199.npy",
                   "nas_bench/nas_bench-x-83.npy",
                   "nas_bench/nas_bench-x-180.npy",
                   "nas_bench/nas_bench-x-176.npy",
                   "nas_bench/nas_bench-x-192.npy",
                   "nas_bench/nas_bench-x-173.npy",
                   "nas_bench/nas_bench-x-41.npy",
                   "nas_bench/nas_bench-x-130.npy",
                   "nas_bench/nas_bench-x-94.npy",
                   "nas_bench/nas_bench-x-56.npy",
                   "nas_bench/nas_bench-x-103.npy",
                   "nas_bench/nas_bench-x-227.npy",
                   "nas_bench/nas_bench-x-219.npy",
                   "nas_bench/nas_bench-x-4.npy",
                   "nas_bench/nas_bench-x-60.npy",
                   "nas_bench/nas_bench-x-203.npy",
                   "nas_bench/nas_bench-x-187.npy",
                   "nas_bench/nas_bench-x-38.npy",
                   "nas_bench/nas_bench-x-52.npy",
                   "nas_bench/nas_bench-x-37.npy",
                   "nas_bench/nas_bench-x-48.npy",
                   "nas_bench/nas_bench-x-127.npy",
                   "nas_bench/nas_bench-x-34.npy",
                   "nas_bench/nas_bench-x-169.npy",
                   "nas_bench/nas_bench-x-114.npy",
                   "nas_bench/nas_bench-x-131.npy",
                   "nas_bench/nas_bench-x-122.npy",
                   "nas_bench/nas_bench-x-92.npy",
                   "nas_bench/nas_bench-x-197.npy",
                   "nas_bench/nas_bench-x-7.npy",
                   "nas_bench/nas_bench-x-234.npy",
                   "nas_bench/nas_bench-x-39.npy",
                   "nas_bench/nas_bench-x-67.npy",
                   "nas_bench/nas_bench-x-254.npy",
                   "nas_bench/nas_bench-x-156.npy",
                   "nas_bench/nas_bench-x-141.npy",
                   "nas_bench/nas_bench-x-185.npy",
                   "nas_bench/nas_bench-x-150.npy",
                   "nas_bench/nas_bench-x-82.npy",
                   "nas_bench/nas_bench-x-68.npy",
                   "nas_bench/nas_bench-x-218.npy",
                   "nas_bench/nas_bench-x-110.npy",
                   "nas_bench/nas_bench-x-36.npy",
                   "nas_bench/nas_bench-x-22.npy",
                   "nas_bench/nas_bench-x-236.npy",
                   "nas_bench/nas_bench-x-118.npy",
                   "nas_bench/nas_bench-x-29.npy",
                   "nas_bench/nas_bench-x-90.npy",
                   "nas_bench/nas_bench-x-208.npy",
                   "nas_bench/nas_bench-x-229.npy",
                   "nas_bench/nas_bench-x-115.npy",
                   "nas_bench/nas_bench-x-17.npy",
                   "nas_bench/nas_bench-x-64.npy",
                   "nas_bench/nas_bench-x-134.npy",
                   "nas_bench/nas_bench-x-251.npy",
                   "nas_bench/nas_bench-x-10.npy",
                   "nas_bench/nas_bench-x-198.npy",
                   "nas_bench/nas_bench-x-237.npy",
                   "nas_bench/nas_bench-x-235.npy",
                   "nas_bench/nas_bench-x-79.npy",
                   "nas_bench/nas_bench-x-9.npy",
                   "nas_bench/nas_bench-x-168.npy",
                   "nas_bench/nas_bench-x-231.npy",
                   "nas_bench/nas_bench-x-152.npy",
                   "nas_bench/nas_bench-x-217.npy",
                   "nas_bench/nas_bench-x-108.npy",
                   "nas_bench/nas_bench-x-42.npy",
                   "nas_bench/nas_bench-x-194.npy",
                   "nas_bench/nas_bench-x-116.npy",
                   "nas_bench/nas_bench-x-104.npy",
                   "nas_bench/nas_bench-x-249.npy",
                   "nas_bench/nas_bench-x-224.npy",
                   "nas_bench/nas_bench-x-206.npy",
                   "nas_bench/nas_bench-x-248.npy",
                   "nas_bench/nas_bench-x-252.npy",
                   "nas_bench/nas_bench-x-49.npy",
                   "nas_bench/nas_bench-x-220.npy",
                   "nas_bench/nas_bench-x-162.npy",
                   "nas_bench/nas_bench-x-182.npy",
                   "nas_bench/nas_bench-x-212.npy",
                   "nas_bench/nas_bench-x-47.npy",
                   "nas_bench/nas_bench-x-240.npy",
                   "nas_bench/nas_bench-x-166.npy",
                   "nas_bench/nas_bench-x-63.npy",
                   "nas_bench/nas_bench-x-93.npy",
                   "nas_bench/nas_bench-x-59.npy",
                   "nas_bench/nas_bench-x-54.npy",
                   "nas_bench/nas_bench-x-43.npy",
                   "nas_bench/nas_bench-x-105.npy",
                   "nas_bench/nas_bench-x-5.npy",
                   "nas_bench/nas_bench-x-69.npy",
                   "nas_bench/nas_bench-x-96.npy",
                   "nas_bench/nas_bench-x-193.npy",
                   "nas_bench/nas_bench-x-20.npy",
                   "nas_bench/nas_bench-x-1.npy",
                   "nas_bench/nas_bench-x-62.npy",
                   "nas_bench/nas_bench-x-87.npy",
                   "nas_bench/nas_bench-x-257.npy",
                   "nas_bench/nas_bench-x-16.npy",
                   "nas_bench/nas_bench-x-233.npy",
                   "nas_bench/nas_bench-x-149.npy",
                   "nas_bench/nas_bench-x-6.npy",
                   "nas_bench/nas_bench-x-151.npy",
                   "nas_bench/nas_bench-x-170.npy",
                   "nas_bench/nas_bench-x-32.npy",
                   "nas_bench/nas_bench-x-163.npy",
                   "nas_bench/nas_bench-x-95.npy",
                   "nas_bench/nas_bench-x-98.npy",
                   "nas_bench/nas_bench-x-228.npy",
                   "nas_bench/nas_bench-x-256.npy",
                   "nas_bench/nas_bench-x-204.npy",
                   "nas_bench/nas_bench-x-65.npy",
                   "nas_bench/nas_bench-x-137.npy",
                   "nas_bench/nas_bench-x-33.npy",
                   "nas_bench/nas_bench-x-160.npy",
                   "nas_bench/nas_bench-x-71.npy",
                   "nas_bench/nas_bench-x-21.npy",
                   "nas_bench/nas_bench-x-123.npy",
                   "nas_bench/nas_bench-x-165.npy",
                   "nas_bench/nas_bench-x-215.npy",
                   "nas_bench/nas_bench-x-167.npy",
                   "nas_bench/nas_bench-x-242.npy",
                   "nas_bench/nas_bench-x-222.npy",
                   "nas_bench/nas_bench-x-61.npy",
                   "nas_bench/nas_bench-x-145.npy",
                   "nas_bench/nas_bench-x-51.npy",
                   "nas_bench/nas_bench-x-174.npy",
                   "nas_bench/nas_bench-x-129.npy",
                   "nas_bench/nas_bench-x-148.npy",
                   "nas_bench/nas_bench-x-25.npy",
                   "nas_bench/nas_bench-x-28.npy",
                   "nas_bench/nas_bench-x-230.npy",
                   "nas_bench/nas_bench-x-74.npy",
                   "nas_bench/nas_bench-x-106.npy",
                   "nas_bench/nas_bench-x-189.npy",
                   "nas_bench/nas_bench-x-241.npy",
                   "nas_bench/nas_bench-x-91.npy",
                   "nas_bench/nas_bench-x-138.npy",
                   "nas_bench/nas_bench-x-15.npy",
                   "nas_bench/nas_bench-x-55.npy",
                   "nas_bench/nas_bench-x-35.npy",
                   "nas_bench/nas_bench-x-81.npy",
                   "nas_bench/nas_bench-x-243.npy",
                   "nas_bench/nas_bench-x-225.npy",
                   "nas_bench/nas_bench-x-13.npy",
                   "nas_bench/nas_bench-x-178.npy",
                   "nas_bench/nas_bench-x-144.npy",
                   "nas_bench/nas_bench-x-135.npy",
                   "nas_bench/nas_bench-x-84.npy",
                   "nas_bench/nas_bench-x-216.npy",
                   "nas_bench/nas_bench-x-239.npy",
                   "nas_bench/nas_bench-x-181.npy",
                   "nas_bench/nas_bench-x-119.npy",
                   "nas_bench/nas_bench-x-142.npy",
                   "nas_bench/nas_bench-x-196.npy",
                   "nas_bench/nas_bench-x-53.npy",
                   "nas_bench/nas_bench-x-147.npy",
                   "nas_bench/nas_bench-x-44.npy",
                   "nas_bench/nas_bench-x-11.npy",
                   "nas_bench/nas_bench-x-158.npy",
                   "nas_bench/nas_bench-x-177.npy",
                   "nas_bench/nas_bench-x-179.npy",
                   "nas_bench/nas_bench-x-223.npy",
                   "nas_bench/nas_bench-x-14.npy",
                   "nas_bench/nas_bench-x-58.npy",
                   "nas_bench/nas_bench-x-112.npy",
                   "nas_bench/nas_bench-x-70.npy",
                   "nas_bench/nas_bench-x-155.npy",
                   "nas_bench/nas_bench-x-88.npy",
                   "nas_bench/nas_bench-x-26.npy",
                   "nas_bench/nas_bench-x-80.npy",
                   "nas_bench/nas_bench-x-97.npy",
                   "nas_bench/nas_bench-x-124.npy",
                   "nas_bench/nas_bench-x-186.npy",
                   "nas_bench/nas_bench-x-171.npy",
                   "nas_bench/nas_bench-x-161.npy",
                   "nas_bench/nas_bench-x-202.npy",
                   "nas_bench/nas_bench-x-159.npy",
                   "nas_bench/nas_bench-x-247.npy",
                   "nas_bench/nas_bench-x-140.npy",
                   "nas_bench/nas_bench-x-86.npy",
                   "nas_bench/nas_bench-x-89.npy",
                   "nas_bench/nas_bench-x-100.npy",
                   "nas_bench/nas_bench-x-238.npy",
                   "nas_bench/nas_bench-x-120.npy",
                   "nas_bench/nas_bench-x-221.npy",
                   "nas_bench/nas_bench-x-0.npy",
                   "nas_bench/nas_bench-x-85.npy",
                   "nas_bench/nas_bench-x-102.npy",
                   "nas_bench/nas_bench-x-113.npy",
                   "nas_bench/nas_bench-x-143.npy",
                   "nas_bench/nas_bench-x-27.npy",
                   "nas_bench/nas_bench-x-8.npy",
                   "nas_bench/nas_bench-x-211.npy",
                   "nas_bench/nas_bench-x-195.npy",
                   "nas_bench/nas_bench-x-23.npy",
                   "nas_bench/nas_bench-x-255.npy",
                   "nas_bench/nas_bench-x-172.npy",
                   "nas_bench/nas_bench-x-136.npy",
                   "nas_bench/nas_bench-x-191.npy",
                   "nas_bench/nas_bench-x-244.npy",
                   "nas_bench/nas_bench-x-207.npy",
                   "nas_bench/nas_bench-x-214.npy",
                   "nas_bench/nas_bench-x-66.npy",
                   "nas_bench/nas_bench-x-45.npy",
                   "nas_bench/nas_bench-x-200.npy",
                   "nas_bench/nas_bench-x-126.npy",
                   "nas_bench/nas_bench-x-133.npy",
                   "nas_bench/nas_bench-x-201.npy",
                   "nas_bench/nas_bench-x-188.npy",
                   "nas_bench/nas_bench-x-76.npy",
                   "nas_bench/nas_bench-x-30.npy",
                   "nas_bench/nas_bench-x-31.npy",
                   "nas_bench/nas_bench-x-101.npy",
                   "nas_bench/nas_bench-x-183.npy",
                   "nas_bench/nas_bench-x-19.npy",
                   "nas_bench/nas_bench-x-139.npy",
                   "nas_bench/nas_bench-x-132.npy",
                   "nas_bench/nas_bench-x-107.npy",
                   "nas_bench/nas_bench-x-184.npy",
                   "nas_bench/nas_bench-x-164.npy",
                   "nas_bench/nas_bench-x-246.npy",
                   "nas_bench/nas_bench-x-232.npy",
                   "nas_bench/nas_bench-x-57.npy",
                   "nas_bench/nas_bench-x-72.npy",
                   "nas_bench/nas_bench-x-2.npy",
                   "nas_bench/nas_bench-x-46.npy",
                   "nas_bench/nas_bench-x-205.npy",
                   "nas_bench/nas_bench-x-121.npy",
                   "nas_bench/nas_bench-x-128.npy",
                   "nas_bench/nas_bench-x-111.npy",
                   "nas_bench/nas_bench-x-210.npy",
                   "nas_bench/nas_bench-x-253.npy",
                   "nas_bench/nas_bench-x-50.npy",
                   "nas_bench/nas_bench-x-153.npy",
                   "nas_bench/nas_bench-x-190.npy"]


class NASBenchDataset(DiscreteDataset):
    """An architecture search dataset that defines a common set of functions
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

    x_shards: Union[np.ndarray,
                    RemoteResource,
                    List[np.ndarray],
                    List[RemoteResource]]
        a list of RemoteResource that should be downloaded before the
        dataset can be loaded and used for model-based optimization

    y_shards: Union[np.ndarray,
                    RemoteResource,
                    List[np.ndarray],
                    List[RemoteResource]]
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
    def register_x_shards():
        """Registers a remote file for download that contains design values
        in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [RemoteResource(
            file, is_absolute=False,
            download_target=f"https://design-bench."
                            f"s3-us-west-1.amazonaws.com/{file}",
            download_method="direct") for file in NAS_BENCH_FILES]

    @staticmethod
    def register_y_shards():
        """Registers a remote file for download that contains prediction
        values in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [RemoteResource(
            file.replace("-x-", "-y-"), is_absolute=False,
            download_target=f"https://design-bench."
                            f"s3-us-west-1.amazonaws.com/"
                            f"{file.replace('-x-', '-y-')}",
            download_method="direct") for file in NAS_BENCH_FILES]

    def __init__(self, soft_interpolation=0.6, **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution

        Arguments:

        soft_interpolation: float
            floating point hyper parameter used when converting design values
            from integers to a floating point representation as logits, which
            interpolates between a uniform and dirac distribution
            1.0 = dirac, 0.0 -> uniform
        **kwargs: dict
            additional keyword arguments which are used to parameterize the
            data set generation process, including which shard files are used
            if multiple sets of data set shard files can be loaded

        """

        # initialize the dataset using the method in the base class
        super(NASBenchDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            is_logits=False, num_classes=11,
            soft_interpolation=soft_interpolation, **kwargs)
