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

        """

        # new dataset that shares statistics with this one
        dataset = DiscreteDataset(
            x_shards, y_shards,
            is_logits=kwargs.get("is_logits", self.is_logits),
            num_classes=kwargs.get("num_classes", self.num_classes),
            soft_interpolation=kwargs.get(
                "soft_interpolation", self.soft_interpolation),
            internal_batch_size=kwargs.get(
                "internal_batch_size", self.internal_batch_size))

        # carry over the sub sampling statistics of the parent
        dataset.dataset_min_percentile = self.dataset_min_percentile
        dataset.dataset_max_percentile = self.dataset_max_percentile
        dataset.dataset_min_output = self.dataset_min_output
        dataset.dataset_max_output = self.dataset_max_output
        dataset.dataset_size = dataset.y.shape[0]

        # carry over the normalize statistics of the parent
        if self.is_normalized_x:
            dataset.is_normalized_x = True
            dataset.x_mean = self.x_mean
            dataset.x_standard_dev = self.x_standard_dev

        # carry over the normalize statistics of the parent
        if self.is_normalized_y:
            dataset.is_normalized_y = True
            dataset.y_mean = self.y_mean
            dataset.y_standard_dev = self.y_standard_dev

        # return the re-built dataset
        dataset.freeze_statistics = True
        return dataset

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

        # take a subset of the sliced arrays
        mask = np.logical_and(y_batch <= self.dataset_max_output,
                              y_batch >= self.dataset_min_output)
        indices = np.where(mask)[0]

        # sub sample the design and prediction values
        x_batch = x_batch[indices] if return_x else None
        y_batch = y_batch[indices] if return_y else None

        # convert the design values from integers to logits
        if self.is_logits and return_x \
                and np.issubdtype(x_batch.dtype, np.integer):
            x_batch = self.to_logits(x_batch)

        # convert the design values from logits to integers
        elif not self.is_logits and return_x \
                and np.issubdtype(x_batch.dtype, np.floating):
            x_batch = self.to_integers(x_batch)

        # normalize the design values in the batch
        if self.is_normalized_x and return_x:
            x_batch = self.normalize_x(x_batch)

        # normalize the prediction values in the batch
        if self.is_normalized_y and return_y:
            y_batch = self.normalize_y(y_batch)

        # return processed batches of designs an predictions
        return (x_batch if return_x else None,
                y_batch if return_y else None)

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
        if not self.is_logits:
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
        if not self.is_logits:
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
