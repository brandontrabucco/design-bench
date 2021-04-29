from design_bench.oracles.oracle_builder import OracleBuilder
import abc


class ApproximateOracle(OracleBuilder, abc.ABC):
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

    is_normalized_y: bool
        a boolean indicator that specifies whether the inputs to the oracle
        score function are expected to be normalized
    is_normalized_x: bool
        a boolean indicator that specifies whether the outputs of the oracle
        score function are expected to be normalized
    is_logits: bool
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

    fit(np.ndarray, np.ndarray):
        a function that accepts a data set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

    """

    @staticmethod
    @abc.abstractmethod
    def fit(dataset, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        raise NotImplementedError

    def __init__(self, dataset: DatasetBuilder,
                 internal_batch_size=32, internal_measurements=1,
                 noise_std=0.0, is_normalized_y=False,
                 is_normalized_x=False, is_logits=None, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
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
        is_normalized_y: bool
            a boolean indicator that specifies whether the inputs to the
            oracle score function are expected to be normalized
        is_normalized_x: bool
            a boolean indicator that specifies whether the outputs of the
            oracle score function are expected to be normalized
        is_logits: bool
            a boolean that specifies whether the oracle score function
            is expecting logits when the dataset is discrete

        """

        self.model = self.fit(dataset, **kwargs)

        # initialize the oracle using the super class
        super(ApproximateOracle, self).__init__(
            dataset, is_batched=True, internal_batch_size=internal_batch_size,
            internal_measurements=internal_measurements,
            noise_std=noise_std, is_normalized_y=is_normalized_y,
            is_normalized_x=is_normalized_x, is_logits=is_logits)

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

        return self.model.predict(x)
