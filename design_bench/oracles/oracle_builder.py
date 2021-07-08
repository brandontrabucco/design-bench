from design_bench.datasets.dataset_builder import DatasetBuilder
from design_bench.datasets.discrete_dataset import DiscreteDataset
import abc
import numpy as np
import math


class OracleBuilder(abc.ABC):
    """An abstract class for managing the ground truth score functions f(x)
    for model-based optimization problems, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    external_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which points to
        the mutable task dataset for a model-based optimization problem

    internal_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which has frozen
        statistics and is used for training the oracle

    is_batched: bool
        a boolean variable that indicates whether the evaluation function
        implemented for a particular oracle is batched, which effects
        the scaling coefficient of its computational cost

    internal_batch_size: int
        an integer representing the number of design values to process
        internally at the same time, if None defaults to the entire
        tensor given to the self.score method
    internal_measurements: int
        an integer representing the number of independent measurements of
        the prediction made by the oracle, which are subsequently
        averaged, and is useful when the oracle is stochastic

    noise_std: float
        the standard deviation of gaussian noise added to the prediction
        values 'y' coming out of the ground truth score function f(x)
        in order to make the optimization problem difficult

    expect_normalized_y: bool
        a boolean indicator that specifies whether the inputs to the oracle
        score function are expected to be normalized
    expect_normalized_x: bool
        a boolean indicator that specifies whether the outputs of the oracle
        score function are expected to be normalized
    expect_logits: bool
        a boolean that specifies whether the oracle score function is
        expecting logits when the dataset is discrete

    Public Methods:

    predict(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    check_input_format(DatasetBuilder) -> bool:
        a function that accepts a list of integers as input and returns true
        when design values 'x' with the shape specified by that list are
        compatible with this class of approximate oracle

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

    """

    @property
    @abc.abstractmethod
    def name(self):
        """Attribute that specifies the name of a machine learning model used
        as the ground truth score function for a model-based optimization
        problem; for example, "random_forest"

        """

        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def check_input_format(cls, dataset):
        """a function that accepts a model-based optimization dataset as input
        and determines whether the provided dataset is compatible with this
        oracle score function (is this oracle a correct one)

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        Returns:

        is_compatible: bool
            a boolean indicator that is true when the specified dataset is
            compatible with this ground truth score function

        """

        raise NotImplementedError

    @abc.abstractmethod
    def protected_predict(self, x, **kwargs):
        """Score function to be implemented by oracle subclasses, where x is
        either a batch of designs if self.is_batched is True or is a
        single design when self._is_batched is False

        Arguments:

        x_batch: np.ndarray
            a batch or single design 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y_batch: np.ndarray
            a batch or single prediction 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        raise NotImplementedError

    def __init__(self, dataset: DatasetBuilder, is_batched=True,
                 internal_batch_size=32, internal_measurements=1,
                 noise_std=0.0, expect_normalized_y=False,
                 expect_normalized_x=False, expect_logits=None,
                 max_samples=None, distribution=None,
                 min_percentile=0.0, max_percentile=100.0):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        is_batched: bool
            a boolean variable that indicates whether the evaluation function
            implemented for a particular oracle is batched, which effects
            the scaling coefficient of its computational cost
        internal_batch_size: int
            an integer representing the number of design values to process
            internally at the same time, if None defaults to the entire
            tensor given to the self.score method
        internal_measurements: int
            an integer representing the number of independent measurements of
            the prediction made by the oracle, which are subsequently
            averaged, and is useful when the oracle is stochastic
        noise_std: float
            the standard deviation of gaussian noise added to the prediction
            values 'y' coming out of the ground truth score function f(x)
            in order to make the optimization problem difficult
        expect_normalized_y: bool
            a boolean indicator that specifies whether the inputs to the
            oracle score function are expected to be normalized
        expect_normalized_x: bool
            a boolean indicator that specifies whether the outputs of the
            oracle score function are expected to be normalized
        expect_logits: bool
            a boolean that specifies whether the oracle score function
            is expecting logits when the dataset is discrete
        max_samples: int
            the maximum number of samples to include in the visible dataset;
            if more than this number of samples would be present, samples
            are randomly removed from the visible dataset
        distribution: Callable[np.ndarray, np.ndarray]
            a function that accepts an array of the ranks of designs as
            input and returns the probability to sample each according to
            a distribution---for example, a geometric distribution
        max_percentile: float
            the percentile between 0 and 100 of prediction values 'y' above
            which are hidden from access by members outside the class
        min_percentile: float
            the percentile between 0 and 100 of prediction values 'y' below
            which are hidden from access by members outside the class

        """

        # check that the optional arguments are valid
        if expect_logits is not None and not \
                isinstance(dataset, DiscreteDataset):
            raise ValueError("is_logits is only defined "
                             "for use with discrete datasets")

        # check the given dataset is compatible with this oracle
        if not self.check_input_format(dataset):
            raise ValueError("the given dataset is not compatible")

        # keep the dataset in case it is needed for normalization
        self.external_dataset = dataset

        # return a new version of the dataset
        self.internal_dataset = dataset.rebuild_dataset(
            dataset.x_shards, dataset.y_shards, dataset.dataset_visible_mask)

        # draw statistics from a fixed distribution
        # this is necessary because self.dataset is mutable
        self.internal_dataset.subsample(max_samples=max_samples,
                                        distribution=distribution,
                                        min_percentile=min_percentile,
                                        max_percentile=max_percentile)

        # ensure the statistics dataset has frozen statistics
        self.internal_dataset.update_x_statistics()
        self.internal_dataset.update_y_statistics()
        self.internal_dataset.freeze_statistics = True

        # attributes that describe the input format
        self.expect_normalized_y = expect_normalized_y
        self.expect_normalized_x = expect_normalized_x
        self.expect_logits = expect_logits

        # attributes that describe how designs are evaluated
        self.is_batched = is_batched
        self.internal_batch_size = internal_batch_size
        self.internal_measurements = internal_measurements

        # attributes that describe model predictions
        self.noise_std = noise_std
        self.num_evaluations = self.external_dataset.dataset_size

    def dataset_to_oracle_x(self, x_batch, dataset=None):
        """Helper function for converting from designs contained in the
        dataset format into a format the oracle is expecting to process,
        such as from integers to logits of a categorical distribution

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, defaults to be self.external_dataset

        Returns:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from the
            format of designs contained in the dataset to the format
            expected by the oracle score function

        """

        # the default source is self.external_dataset
        if dataset is None:
            dataset = self.external_dataset

        # handle when the dataset is normalized and the normalization
        # statistics expected by the oracle are different
        if dataset.is_normalized_x:
            x_batch = dataset.denormalize_x(x_batch)

        # handle when the inputs are currently encoded as logits
        # and the oracle may use a different set of conversion parameters
        if isinstance(dataset, DiscreteDataset) and dataset.is_logits:
            x_batch = dataset.to_integers(x_batch)

        # handle when the oracle expects logits but the dataset
        # is currently encoded as integers
        if isinstance(dataset, DiscreteDataset) and self.expect_logits:
            x_batch = self.internal_dataset.to_logits(x_batch)

        # handle when the oracle expects normalized designs but
        # the dataset is not currently normalized
        if self.expect_normalized_x:
            x_batch = self.internal_dataset.normalize_x(x_batch)

        return x_batch

    def dataset_to_oracle_y(self, y_batch, dataset=None):
        """Helper function for converting from predictions contained in the
        dataset format into a format the oracle is expecting to process,
        such as from normalized to denormalized predictions

        Arguments:

        y_batch: np.ndarray
            a batch of prediction values 'y' that are from the dataset and
            will be processed into a format expected by the oracle score
            function, which is useful when training the oracle
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, defaults to be self.external_dataset

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of predictions contained in the dataset to the
            format expected by the oracle score function

        """

        # the default source is self.external_dataset
        if dataset is None:
            dataset = self.external_dataset

        # handle when the dataset is normalized and the normalization
        # statistics expected by the oracle are different
        if dataset.is_normalized_y:
            y_batch = dataset.denormalize_y(y_batch)

        # handle when the oracle expects normalized predictions but
        # the dataset is not currently normalized
        if self.expect_normalized_y:
            y_batch = self.internal_dataset.normalize_y(y_batch)

        return y_batch

    def oracle_to_dataset_x(self, x_batch, dataset=None):
        """Helper function for converting from designs in the format of the
        oracle into the design format the dataset contains, such as
        from categorical logits to integers

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from
            the format of designs contained in the dataset to the
            format expected by the oracle score function
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the destination of the batch, defaults to be self.external_dataset

        Returns:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from
            the format of the oracle to the format of designs
            contained in the dataset

        """

        # the default source is self.external_dataset
        if dataset is None:
            dataset = self.external_dataset

        # handle when the dataset is normalized and the normalization
        # statistics expected by the oracle are different
        if self.expect_normalized_x:
            x_batch = self.internal_dataset.denormalize_x(x_batch)

        # handle when the inputs are currently encoded as logits
        # and the oracle may use a different set of conversion parameters
        if isinstance(dataset, DiscreteDataset) and self.expect_logits:
            x_batch = self.internal_dataset.to_integers(x_batch)

        # handle when the oracle expects logits but the dataset
        # is currently encoded as integers
        if isinstance(dataset, DiscreteDataset) and dataset.is_logits:
            x_batch = dataset.to_logits(x_batch)

        # handle when the oracle expects normalized designs but
        # the dataset is not currently normalized
        if dataset.is_normalized_x:
            x_batch = dataset.normalize_x(x_batch)

        return x_batch

    def oracle_to_dataset_y(self, y_batch, dataset=None):
        """Helper function for converting from predictions in the
        format of the oracle into a format the dataset contains,
        such as from normalized to denormalized predictions

        Arguments:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of predictions contained in the dataset to the
            format expected by the oracle score function
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the destination of the batch, defaults to be self.external_dataset

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of the oracle to the format of predictions
            contained in the dataset

        """

        # the default source is self.dataset
        if dataset is None:
            dataset = self.external_dataset

        # handle when the oracle expects normalized predictions but
        # the dataset is not currently normalized
        if self.expect_normalized_y:
            y_batch = self.internal_dataset.denormalize_y(y_batch)

        # handle when the dataset is normalized and the normalization
        # statistics expected by the oracle are different
        if dataset.is_normalized_y:
            y_batch = dataset.normalize_y(y_batch)

        return y_batch

    def predict(self, x_batch, dataset=None, **kwargs):
        """a function that accepts a batch of design values 'x' as input and
        for each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, defaults to be self.external_dataset

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        # calculate the batch size for evaluation
        batch_size = (self.internal_batch_size if
                      self.internal_batch_size is not None
                      else x_batch.shape[0]) if self.is_batched else 1

        # a list to store new predictions
        y_batch = []

        # iterate through all possible read positions in x_batch
        for read_position in range(int(
                math.ceil(x_batch.shape[0] / batch_size))):
            read_position *= batch_size

            # slice out a batch_size portion of x_batch
            x_sliced = x_batch[read_position:read_position + batch_size]

            # convert from the dataset format to the oracle format
            x_sliced = self.dataset_to_oracle_x(x_sliced, dataset=dataset)

            # if the inner score function is not batched squeeze the
            # outermost batch dimension of one
            if not self.is_batched:
                x_sliced = np.squeeze(x_sliced, 0)

            # take multiple independent measurements of the score
            y_sliced = np.mean([self.protected_predict(
                x_sliced, **kwargs) for _ in
                range(self.internal_measurements)], axis=0)

            # if the inner score function is nto batched then add back
            # an outermost batch dimension of one
            if not self.is_batched:
                y_sliced = y_sliced[np.newaxis]

            # potentially add gaussian random noise to the score
            # to save on computation only do this when the std is > 0
            if self.noise_std > 0:
                y_sliced += self.noise_std * np.random.normal(
                    0.0, 1.0, y_sliced.shape).astype(y_sliced.dtype)

            # convert from the oracle format to the dataset format
            y_sliced = self.oracle_to_dataset_y(y_sliced, dataset=dataset)

            # store the new prediction in a list
            y_batch.append(y_sliced)

        # join together all the predictions made so far
        y_batch = np.concatenate(y_batch, axis=0)

        # increment the total number of predictions made by the oracle
        self.num_evaluations += y_batch.shape[0]
        return y_batch
