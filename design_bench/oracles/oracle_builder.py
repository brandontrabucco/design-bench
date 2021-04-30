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
        a boolean indicator that specifies whether the inputs to the oracle
        score function are expected to be normalized
    expect_normalized_x: bool
        a boolean indicator that specifies whether the outputs of the oracle
        score function are expected to be normalized
    expect_logits: bool
        a boolean that specifies whether the oracle score function is
        expecting logits when the dataset is discrete

    Public Methods:

    score(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    check_input_format(DatasetBuilder) -> bool:
        a function that accepts a list of integers as input and returns true
        when design values 'x' with the shape specified by that list are
        compatible with this class of approximate oracle

    """

    @abc.abstractmethod
    def check_input_format(self, dataset):
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
    def protected_score(self, x):
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
                 internal_batch_size=None, internal_measurements=1,
                 noise_std=0.0, expect_normalized_y=False,
                 expect_normalized_x=False, expect_logits=None):
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
        self.dataset = dataset

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
        self.num_evaluations = self.dataset.dataset_size

    def score(self, x_batch):
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

        # calculate the batch size for evaluation
        batch_size = (self.internal_batch_size if
                      self.internal_batch_size is not None
                      else x_batch.shape[0]) if self.is_batched else 1

        # a list to store new predictions
        y_batch = []

        # iterate through all possible read positions in x_batch
        for read_position in range(0, int(
                math.ceil(x_batch.shape[0] / batch_size)), batch_size):

            # slice out a batch_size portion of x_batch
            x_sliced = x_batch[read_position:read_position + batch_size]

            # if the inner score function is not batched squeeze the
            # outermost batch dimension of one
            if not self.is_batched:
                x_sliced = np.squeeze(x_sliced, 0)

            # handle when the oracle expects logits but the dataset
            # is currently encoded as integers
            if isinstance(self.dataset, DiscreteDataset) and \
                    self.expect_logits and not self.dataset.is_logits:
                x_sliced = self.dataset.to_logits(x_sliced)

            # handle when the oracle expects integers but the dataset
            # is currently encoded as logits
            if isinstance(self.dataset, DiscreteDataset) and \
                    not self.expect_logits and self.dataset.is_logits:
                x_sliced = self.dataset.to_integers(x_sliced)

            # handle when the oracle expects normalized designs but
            # the dataset is currently not normalized
            if self.expect_normalized_x and \
                    not self.dataset.is_normalized_x:
                x_sliced = self.dataset.normalize_x(x_sliced)

            # handle when the oracle expects denormalized designs but
            # the dataset is currently normalized
            if not self.expect_normalized_x and \
                    self.dataset.is_normalized_x:
                x_sliced = self.dataset.denormalize_x(x_sliced)

            # take multiple independent measurements of the score
            y_sliced = np.mean([self.protected_score(x_sliced) for _ in
                                range(self.internal_measurements)], axis=0)

            # handle when the dataset expects normalized predictions but
            # the oracle is currently not normalized
            if self.expect_normalized_y and \
                    not self.dataset.is_normalized_y:
                y_sliced = self.dataset.denormalize_y(y_sliced)

            # handle when the dataset expects denormalized designs but
            # the oracle is currently normalized
            if not self.expect_normalized_y and \
                    self.dataset.is_normalized_y:
                y_sliced = self.dataset.normalize_y(y_sliced)

            # if the inner score function is nto batched then add back
            # an outermost batch dimension of one
            if not self.is_batched:
                y_sliced = y_sliced[np.newaxis]

            # store the new prediction in a list
            y_batch.append(y_sliced)

        # join together all the predictions made so far
        y_batch = np.concatenate(y_batch, axis=0)

        # increment the total number of predictions made by the oracle
        self.num_evaluations += y_batch.shape[0]

        # potentially add gaussian random noise to the score
        noise = 0 if self.noise_std == 0 else \
            np.random.normal(0.0, self.noise_std,
                             y_batch.shape).astype(y_batch.dtype)
        return y_batch + noise
