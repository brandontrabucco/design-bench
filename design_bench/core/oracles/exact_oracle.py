from design_bench.core.oracles.oracle_builder import OracleBuilder
import abc


class ExactOracle(OracleBuilder, abc.ABC):
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

    correct_set: set of DatasetBuilder subclasses
        a set of python classes that inherit from DatasetBuilder but have a
        defined "load_dataset" instance method, for which this exact oracle
        is a correct ground truth score function f(x)

    Public Methods:

    score(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    estimate_cost(int) -> float:
        a function that accepts a positive integer as input and uses a linear
        regression model to estimate the computational cost in seconds
        for making that many predictions using this oracle model

    """

    @property
    @abc.abstractmethod
    def correct_set(self):
        """a set of python classes that inherit from DatasetBuilder but have a
        defined "load_dataset" instance method, for which this exact oracle
        is a correct ground truth score function f(x)

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

        # check if the given dataset is in the supported set
        if not any([isinstance(
                dataset, cls) for cls in self.correct_set]):
            raise ValueError("oracle cannot be used with unsupported dataset")

        # initialize using the superclass method
        super(ExactOracle, self).__init__(
            dataset, sample_designs=sample_designs,
            is_batched=is_batched, noise_std=noise_std, **kwargs)
