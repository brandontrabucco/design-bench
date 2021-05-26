from design_bench.oracles.exact_oracle import ExactOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.datasets.discrete.nas_bench_dataset import NASBenchDataset
import numpy as np


class NASBenchOracle(ExactOracle):
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

    """

    name = "exact_cifar_10_test_accuracy"

    @classmethod
    def supported_datasets(cls):
        """An attribute the defines the set of dataset classes which this
        oracle can be applied to forming a valid ground truth score
        function for a model-based optimization problem

        """

        return {NASBenchDataset}

    @classmethod
    def fully_characterized(cls):
        """An attribute the defines whether all possible inputs to the
        model-based optimization problem have been evaluated and
        are are returned via lookup in self.predict

        """

        return True

    @classmethod
    def is_simulated(cls):
        """An attribute the defines whether the values returned by the oracle
         were obtained by running a computer simulation rather than
         performing physical experiments with real data

        """

        return True

    def protected_predict(self, x):
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

        x_key = tuple(x.tolist())
        return self.sequence_to_score[x_key].astype(np.float32) \
            if x_key in self.sequence_to_score else np.full(
            [1], self.internal_dataset.dataset_min_output, dtype=np.float32)

    def __init__(self, dataset: DiscreteDataset, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DiscreteDataset
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        # initialize the oracle using the super class
        super(NASBenchOracle, self).__init__(
            dataset, is_batched=False,
            internal_batch_size=1, internal_measurements=1,
            expect_normalized_y=False,
            expect_normalized_x=False, expect_logits=False, **kwargs)

        # dictionary containing every point in the search space
        self.sequence_to_score = dict()
        self.internal_dataset._disable_transform = True
        for x, y in self.internal_dataset.iterate_samples():
            self.sequence_to_score[tuple(x.tolist())] = y
        self.internal_dataset._disable_transform = False
