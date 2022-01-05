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

    name = "exact_average_return"

    @classmethod
    def supported_datasets(cls):
        """An attribute the defines the set of dataset classes which this
        oracle can be applied to forming a valid ground truth score
        function for a model-based optimization problem

        """

        return {HopperControllerDataset}

    @classmethod
    def fully_characterized(cls):
        """An attribute the defines whether all possible inputs to the
        model-based optimization problem have been evaluated and
        are are returned via lookup in self.predict

        """

        return False

    @classmethod
    def is_simulated(cls):
        """An attribute the defines whether the values returned by the oracle
         were obtained by running a computer simulation rather than
         performing physical experiments with real data

        """

        return True

    def protected_predict(self, x, render=False, **render_kwargs):
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
                  (1, self.hidden_dim,),
                  (self.hidden_dim, self.hidden_dim),
                  (1, self.hidden_dim,),
                  (self.hidden_dim, self.action_dim),
                  (1, self.action_dim,),
                  (1, self.action_dim)):
            weights.append(x[0:np.prod(s)].reshape(s))
            x = x[np.prod(s):]

        # create a policy forward pass in numpy
        def mlp_policy(h):
            h = h.reshape(1, -1)
            h = np.tanh(h @ weights[0] + weights[1])
            h = np.tanh(h @ weights[2] + weights[3])
            h = h @ weights[4] + weights[5] + np.random.randn(1, self.action_dim) * np.exp(weights[6])
            return h

        # make a copy of the policy and the environment
        env = gym.make(self.env_name)

        # perform a single rollout for quick evaluation
        path_returns = np.zeros([1], dtype=np.float32)
        total_return = 0.0
        for _ in range(self.eval_n_trials):
            obs = env.reset()
            done = False
            for step in range(1000):
                obs, rew, done, info = env.step(mlp_policy(obs))
                if render:
                    env.render(**render_kwargs)
                total_return += rew
                if done:
                    break
        path_returns[0] = total_return / self.eval_n_trials

        # return the sum of rewards for a single trajectory
        return path_returns.astype(np.float32)

    def __init__(self, dataset: ContinuousDataset, **kwargs):
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
        self.eval_n_trials = 10

        # initialize the oracle using the super class
        super(HopperControllerOracle, self).__init__(
            dataset, internal_batch_size=1, is_batched=False,
            expect_normalized_y=False,
            expect_normalized_x=False, expect_logits=None, **kwargs)
