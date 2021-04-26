from sklearn.linear_model import LinearRegression
import abc
import time
import numpy as np


class OracleBuilder(abc.ABC):
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

    """

    @abc.abstractmethod
    def load_oracle(self, dataset, **kwargs):
        """A function that load saved parameters of the ground truth
        score function f(x) such that the score function can be queried
        on candidate solutions to the model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            the shapes and data types of those attributes
        **kwargs: dict
            additional keyword arguments passed to the "load_oracle" method,
            which may be task specific and depend on what class of model
            or simulator is being used as the ground truth score function

        """

        raise NotImplementedError

    @abc.abstractmethod
    def protected_score(self, x):
        """a function that accepts a batch of design values 'x' as input and
        for each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

        Arguments:

        x: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y: np.ndarray
            a batch of prediction values 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

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

        # set up preliminary class attributes
        self.dataset = dataset
        self.is_batched = is_batched
        self.noise_std = noise_std

        # load the oracle model and its parameters as class attributes
        self.load_oracle(dataset, **kwargs)

        # set up initial computational cost attributes
        self.evaluation_cost = None
        self.scaling_coefficient = None
        self.num_evaluations = 0

        # if any sample designs were provided that calculate an estimated
        # computational cost for running the oracle model
        if sample_designs is not None:

            # loop through different sizes of batches of designs
            evaluation_times = []
            for i in range(sample_designs.shape[0]):

                # select a batch of designs fo size i + 1
                x = sample_designs[:i + 1]

                # time how long it takes in seconds to get a score
                start_time = time.time()
                self.protected_score(x)
                end_time = time.time()

                # calculate the time elapsed and record it
                evaluation_times.append(end_time - start_time)

            # create a small data set to fit a linear model
            x = np.arange(sample_designs.shape[0]).reshape((-1, 1))
            y = np.array(evaluation_times)

            # perform linear regression on the computation cost
            cost_model = LinearRegression()
            cost_model.fit(x, y)

            # calculate the evaluation cost and scaling coefficient
            self.evaluation_cost = cost_model.intercept_
            self.scaling_coefficient = cost_model.coef_

    def score(self, x):
        """a function that accepts a batch of design values 'x' as input and
        for each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

        Arguments:

        x: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y: np.ndarray
            a batch of prediction values 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        # increment the total number of function evaluations
        self.num_evaluations += x.shape[0]

        # compute the score of a design 'x' and add noise
        y = self.protected_score(x)
        return y + self.noise_std * \
            np.random.normal(0.0, 1.0, y.shape).astype(y.dtype)

    def estimate_cost(self, num_x):
        """a function that accepts a positive integer as input and uses a
        linear regression model to estimate the computational cost in
        seconds for making that many predictions using this oracle model

        Arguments:

        num_x: int
            the number of design values 'x' for which the goal is to estimate
            the total computational cost in seconds if these designs were
            given to the oracle model as as input

        Returns:

        cost: float
            the estimated computational cost in seconds when a total of 'num_x'
            randomly sampled design values 'x' are given to the oracle model
            as input before a prediction value 'y' is obtained

        """

        return self.evaluation_cost + \
            (float(num_x) - 1) * self.scaling_coefficient
