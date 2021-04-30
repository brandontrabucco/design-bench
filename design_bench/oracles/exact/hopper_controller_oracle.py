from design_bench.oracles.exact_oracle import ExactOracle
from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset
import numpy as np
import gym


class HopperControllerOracle(ExactOracle):
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

        return isinstance(dataset, HopperControllerDataset)

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

        # extract weights from the vector design
        weights = []
        for s in ((self.obs_dim, self.hidden_dim),
                  (self.hidden_dim,),
                  (self.hidden_dim, self.hidden_dim),
                  (self.hidden_dim,),
                  (self.hidden_dim, self.action_dim),
                  (self.action_dim,),
                  (1, self.action_dim)):
            weights.append(x[0:np.prod(s)].reshape(s))
            x = x[np.prod(s):]

        # the final weight is logstd and is not used
        weights.pop(-1)

        # create a policy forward pass in numpy
        def mlp_policy(h):
            h = np.tanh(h @ weights[0] + weights[1])
            h = np.tanh(h @ weights[2] + weights[3])
            return h @ weights[4] + weights[5]

        # make a copy of the policy and the environment
        env = gym.make(self.env_name)

        # perform a single rollout for quick evaluation
        obs = env.reset()
        done = False
        path_returns = np.zeros([1], dtype=np.float32)
        while not done:
            obs, rew, done, info = env.step(mlp_policy(obs))
            path_returns += rew.astype(np.float32)

        # return the sum of rewards for a single trajectory
        return path_returns

    def __init__(self, dataset: ContinuousDataset,
                 noise_std=0.0, internal_measurements=1):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DiscreteDataset
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        noise_std: float
            the standard deviation of gaussian noise added to the prediction
            values 'y' coming out of the ground truth score function f(x)
            in order to make the optimization problem difficult
        internal_measurements: int
            an integer representing the number of independent measurements of
            the prediction made by the oracle, which are subsequently
            averaged, and is useful when the oracle is stochastic

        """

        self.obs_dim = 11
        self.action_dim = 3
        self.hidden_dim = 64
        self.env_name = 'Hopper-v2'

        # initialize the oracle using the super class
        super(HopperControllerOracle, self).__init__(
            dataset, noise_std=noise_std, internal_batch_size=1,
            is_batched=False, internal_measurements=internal_measurements,
            expect_normalized_y=False,
            expect_normalized_x=False, expect_logits=None)
