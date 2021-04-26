from design_bench.core.oracles.oracle_builder import OracleBuilder
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
        the shapes and data types of those attributes

    noise_std: float
        the standard deviation of gaussian noise added to the prediction
        values 'y' coming out of the ground truth score function f(x)
        in order to make the optimization problem difficult

    num_evaluations: int
        the number of evaluations made by this oracle represented as an
        integer, which is recorded so the sample-efficiency of a
        model-based optimization algorithm can be tracked

    evaluation_cost: float
        the approximate time in seconds for a single evaluation to be
        performed, which is useful when estimating the computational
        cost of a model-based optimization algorithm

    scaling_coefficient: float
        the scaling coefficient of a linear approximation to the function
        that relates the number of design values being evaluated to
        the amount of time taken during evaluation

    is_batched: bool
        a boolean variable that indicates whether the evaluation function
        implemented for a particular oracle is batched, which effects
        the scaling coefficient of its computational cost

    Public Methods:

    score(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    estimate_cost(int) -> float:
        a function that accepts a positive integer as input and uses a linear
        regression model to estimate the computational cost in seconds
        for making that many predictions using this oracle model

    fit(np.ndarray, np.ndarray):
        a function that accepts a data set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

    check_input_format(list of int) -> bool:
        a function that accepts a list of integers as input and returns true
        when design values 'x' with the shape specified by that list are
        compatible with this class of approximate oracle

    """

    @abc.abstractmethod
    def check_input_format(self, shape, dtype):
        """a function that accepts a list of integers as input and is true
        when design values 'x' with the shape specified by that list are
        compatible with this class of approximate oracle

        Arguments:

        shape: list of int
            the shape of a single design value 'x' without a batch dimension
            that would be passed to the ground truth score function
        dtype: np.dtype
            the data type (such as np.float32) of the design values 'x' that
            will be passed to the ground truth score function

        Returns:

        is_compatible: bool
            a boolean indicator that is true when the specified input format
            is compatible with this ground truth score function

        """

        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x, y):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        x: np.ndarray
            the design values 'x' for a model-based optimization problem
            represented as a numpy array of arbitrary type
        y: np.ndarray
            the prediction values 'y' for a model-based optimization problem
            represented by a scalar floating point value per 'x'

        """

        raise NotImplementedError

    def __init__(self, dataset, sample_designs=None,
                 is_batched=False, noise_std=0.0, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            the shapes and data types of those attributes
        sample_designs: np.ndarray
            a batch of randomly selected design values 'x' from the
            distribution on which this oracle will be used to compute
            prediction values 'y' during testing
        is_batched: bool
            a boolean variable that indicates whether the evaluation function
            implemented for a particular oracle is batched, which effects
            the scaling coefficient of its computational cost
        noise_std: float
            the standard deviation of gaussian noise added to the prediction
            values 'y' coming out of the ground truth score function f(x)
            in order to make the optimization problem difficult
        **kwargs: dict
            additional keyword arguments passed to the "load_oracle" method,
            which may be task specific and depend on what class of model
            or simulator is being used as the ground truth score function

        """

        # check if the given dataset matches the supported criteria
        if not self.check_input_format(
                dataset.input_shape, dataset.input_dtype):
            raise ValueError("oracle cannot be used with unsupported dataset")

        # initialize using the superclass method
        super(ApproximateOracle, self).__init__(
            dataset, sample_designs=sample_designs,
            is_batched=is_batched, noise_std=noise_std, **kwargs)
