from design_bench.datasets.dataset_builder import DatasetBuilder
import numpy as np


def one_hot(a, num_classes):
    """A helper function that converts integers into a floating
    point one-hot representation using pure numpy:
    https://stackoverflow.com/questions/36960320/
    convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy

    """

    out = np.zeros((a.size, num_classes), dtype=np.float32)
    out[np.arange(a.size), a.ravel()] = 1.0
    out.shape = a.shape + (num_classes,)
    return out


class DiscreteDataset(DatasetBuilder):
    """Discrete dataset base class that defines a common set of functions
    and attributes for a model-based optimization dataset, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    name: str
        An attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench
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

    internal_batch_size: int
        the integer number of samples per batch that is used internally
        when processing the dataset and generating samples
    freeze_statistics: bool
        a boolean indicator that when set to true prevents methods from
        changing the normalization and sub sampling statistics

    is_normalized_x: bool
        a boolean indicator that specifies whether the design values
        in the dataset are being normalized
    x_mean: np.ndarray
        a numpy array that is automatically calculated to be the mean
        of visible design values in the dataset
    x_standard_dev: np.ndarray
        a numpy array that is automatically calculated to be the standard
        deviation of visible design values in the dataset

    is_normalized_y: bool
        a boolean indicator that specifies whether the prediction values
        in the dataset are being normalized
    y_mean: np.ndarray
        a numpy array that is automatically calculated to be the mean
        of visible prediction values in the dataset
    y_standard_dev: np.ndarray
        a numpy array that is automatically calculated to be the standard
        deviation of visible prediction values in the dataset

    is_logits: bool
        a value that indicates whether the design values contained in the
        model-based optimization dataset have already been converted to
        logits and need not be converted again
    num_classes: int
        an integer representing the number of classes in the distribution
        that the integer data points are sampled from which cannot be None
        and must also be greater than 1

    Public Methods:

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

    subsample(max_samples: int,
              max_percentile: float,
              min_percentile: float):
        a function that exposes a subsampled version of a much larger
        model-based optimization dataset containing design values 'x'
        whose prediction values 'y' are skewed
    relabel(relabel_function:
            Callable[[np.ndarray, np.ndarray], np.ndarray]):
        a function that accepts a function that maps from a dataset of
        design values 'x' and prediction values y to a new set of
        prediction values 'y' and relabels the model-based optimization dataset

    clone(subset: set, shard_size: int,
          to_disk: bool, disk_target: str, is_absolute: bool):
        Generate a cloned copy of a model-based optimization dataset
        using the provided name and shard generation settings; useful
        when relabelling a dataset buffer from the dis
    split(fraction: float, subset: set, shard_size: int,
          to_disk: bool, disk_target: str, is_absolute: bool):
        split a model-based optimization data set into a training set and
        a validation set allocating 'fraction' of the data set to the
        validation set and the rest to the training set

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

    name = "DiscreteDataset"
    x_name = "Design"
    y_name = "Prediction"

    @property
    def subclass_kwargs(self):
        """Generate a dictionary containing class initialization keyword
        arguments that are specific to sub classes; for example, may contain
        the number of classes in a discrete dataset

        """

        return dict(is_logits=self.is_logits, num_classes=self.num_classes,
                    soft_interpolation=self.soft_interpolation)

    @property
    def subclass(self):
        """Specifies the primary subclass of an instance of DatasetBuilder
        that can be instantiated on its own using self.rebuild_dataset
        and typically either DiscreteDataset or ContinuousDataset

        """

        return DiscreteDataset

    def __init__(self, *args, is_logits=False,
                 num_classes=2, soft_interpolation=0.6, **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution

        Arguments:

        *args: list
            a list of positional arguments passed to the super class
            constructor of the DiscreteDataset class, which typically
            includes a list of x shards and y shards; see dataset_builder.py
        is_logits: bool
            a value that indicates whether the design values contained in the
            model-based optimization dataset have already been converted to
            logits and need not be converted again
        num_classes: int
            an integer representing the number of classes in the distribution
            that the integer data points are sampled from which cannot be None
            and must also be greater than 1
        soft_interpolation: float
            a floating point hyper parameter used when converting design values
            from integers to a floating point representation as logits, which
            interpolates between a uniform and dirac distribution
            1.0 = dirac, 0.0 -> uniform
        **kwargs: dict
            additional keyword arguments which are used to parameterize the
            data set generation process, including which shard files are used
            if multiple sets of data set shard files can be loaded

        """

        # set a hyper parameter that controls the conversion from
        # integers to floating point logits for the dataset
        self.soft_interpolation = soft_interpolation
        self.num_classes = num_classes
        self.is_logits = is_logits

        # initialize the dataset using the method in the base class
        super(DiscreteDataset, self).__init__(*args, **kwargs)

    def batch_transform(self, x_batch, y_batch,
                        return_x=True, return_y=True):
        """Apply a transformation to batches of samples from a model-based
        optimization data set, including sub sampling and normalization
        and potentially other used defined transformations

        Arguments:

        x_batch: np.ndarray
            a numpy array representing a batch of design values sampled
            from a model-based optimization data set
        y_batch: np.ndarray
            a numpy array representing a batch of prediction values sampled
            from a model-based optimization data set
        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True

        Returns:

        x_batch: np.ndarray
            a numpy array representing a batch of design values sampled
            from a model-based optimization data set
        y_batch: np.ndarray
            a numpy array representing a batch of prediction values sampled
            from a model-based optimization data set

        """

        # convert the design values from integers to logits
        if self.is_logits and return_x \
                and np.issubdtype(x_batch.dtype, np.integer):
            x_batch = self.to_logits(x_batch)

        # convert the design values from logits to integers
        elif not self.is_logits and return_x \
                and np.issubdtype(x_batch.dtype, np.floating):
            x_batch = self.to_integers(x_batch)

        # return processed batches of designs an predictions
        return super(DiscreteDataset, self).batch_transform(
            x_batch, y_batch, return_x=return_x, return_y=return_y)

    def update_x_statistics(self):
        """A helpful function that calculates the mean and standard deviation
        of the designs and predictions in a model-based optimization dataset
        either iteratively or all at once using numpy

        """

        # handle corner case when we need statistics but they were
        # not computed yet and the dataset is currently mapped to integers
        original_is_logits = self.is_logits
        self.is_logits = True
        super(DiscreteDataset, self).update_x_statistics()
        self.is_logits = original_is_logits

    def rebuild_dataset(self, x_shards, y_shards, visible_mask):
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

        visible_mask: np.ndarray
            a numpy array of shape [dataset_size] containing boolean entries
            specifying which samples are visible in the provided Iterable

        Returns:

        dataset: DatasetBuilder
            an instance of a data set builder subclass containing a copy
            of all statistics associated with this dataset

        """

        # build the dataset using the super class method
        dataset = super(DiscreteDataset, self)\
            .rebuild_dataset(x_shards, y_shards, visible_mask)

        # carry over the shape and the data type of the designs
        dataset.input_shape = self.input_shape
        dataset.input_size = self.input_size
        dataset.input_dtype = self.input_dtype

        # potentially convert the dataset from integers to logits
        dataset.is_logits = self.is_logits
        return dataset

    def map_normalize_x(self):
        """a function that standardizes the design values 'x' to have zero
        empirical mean and unit empirical variance in the dataset

        """

        # check that the dataset is in a form that supports normalization
        if not self.is_logits:
            raise ValueError("cannot normalize discrete design values")

        # call the normalization method of the super class
        super(DiscreteDataset, self).map_normalize_x()

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

        # check that the dataset is in a form that supports normalization
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot normalize discrete design values")

        # call the normalization method of the super class
        return super(DiscreteDataset, self).normalize_x(x)

    def map_denormalize_x(self):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance in the dataset

        """

        # check that the dataset is in a form that supports denormalization
        if not self.is_logits:
            raise ValueError("cannot denormalize discrete design values")

        # call the normalization method of the super class
        super(DiscreteDataset, self).map_denormalize_x()

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

        # check that the dataset is in a form that supports denormalization
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot denormalize discrete design values")

        # call the normalization method of the super class
        return super(DiscreteDataset, self).denormalize_x(x)

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

        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("cannot convert non-integers to logits")

        # convert the integers to one hot vectors
        one_hot_x = one_hot(x, self.num_classes)

        # build a uniform distribution to interpolate between
        uniform_prior = np.full_like(one_hot_x, 1 / float(self.num_classes))

        # interpolate between a dirac distribution and a uniform prior
        soft_x = self.soft_interpolation * one_hot_x + (
            1.0 - self.soft_interpolation) * uniform_prior

        # convert to log probabilities
        x = np.log(soft_x)

        # remove one degree of freedom caused by \sum_i p_i = 1.0
        return (x[:, :, 1:] - x[:, :, :1]).astype(np.float32)

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

        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot convert non-floats to integers")

        # add an additional component of zero and find the class
        # with maximum probability
        return np.argmax(np.pad(x, [[0, 0]] * (
            len(x.shape) - 1) + [[1, 0]]), axis=-1).astype(np.int32)

    def map_to_logits(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if not self.is_logits:

            # set the appropriate state variable
            self.is_logits = True

            # check shape and data type of a single design value x
            for x0 in self.iterate_samples(return_y=False):
                self.input_shape = x0.shape
                self.input_size = int(np.prod(x0.shape))
                self.input_dtype = x0.dtype
                break

    def map_to_integers(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if self.is_logits:

            # if design values are normalized then denorm
            if self.is_normalized_x:
                self.map_denormalize_x()

            # set the appropriate state variable
            self.is_logits = False

            # check shape and data type of a single design value x
            for x0 in self.iterate_samples(return_y=False):
                self.input_shape = x0.shape
                self.input_size = int(np.prod(x0.shape))
                self.input_dtype = x0.dtype
                break
