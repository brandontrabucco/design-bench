from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
import numpy as np
import gym
import os


class ControllerV0Task(Task):

    def score(self, x):
        return NotImplemented

    def __init__(self,
                 obs_dim=11,
                 action_dim=3,
                 hidden_dim=64,
                 env_name='Hopper-v2',
                 x_file='hopper_controller_v0_X.npy',
                 y_file='hopper_controller_v0_y.npy',
                 split_percentile=100,
                 ys_noise=0.0):
        """Load static datasets of weights and their corresponding
        expected returns from the disk

        Args:

        obs_dim: int
            the number of channels in the environment observations
        action_dim: int
            the number of channels in the environment actions
        hidden_dim: int
            the number of channels in policy hidden layers
        env_name: str
            the name of the gym.Env to use when collecting rollouts
        x_file: str
            the name of the dataset file to be loaded for x
        y_file: str
            the name of the dataset file to be loaded for y
        split_percentile: int
            the percentile (out of 100) to split the data set by and only
            include samples with score below this percentile
        ys_noise: float
            the number of standard deviations of noise to add to
            the static training dataset y values accompanying this task
        """

        maybe_download('1U997qfr5ZUNPFlC29jxdPjA42xCiohaV',
                       os.path.join(DATA_DIR, 'hopper_controller_v0_X.npy'))
        maybe_download('1AQmCaerm1gmlJdajBZm0JAsKjbAbR3r8',
                       os.path.join(DATA_DIR, 'hopper_controller_v0_y.npy'))

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.env_name = env_name

        x = np.load(os.path.join(DATA_DIR, x_file))
        y = np.load(os.path.join(DATA_DIR, y_file))
        x = x.astype(np.float32)
        y = y.astype(np.float32).reshape([-1, 1])

        split_value = np.percentile(y[:, 0], split_percentile)
        indices = np.where(y <= split_value)[0]
        y = y[indices]
        x = x[indices]

        mean_y = np.mean(y, axis=0, keepdims=True)
        st_y = np.std(y - mean_y, axis=0, keepdims=True)
        y = y + np.random.normal(0.0, 1.0, y.shape) * st_y * ys_noise

        self.x = x
        self.y = y
        self.score = np.vectorize(self.scalar_score,
                                  signature='(n)->(1)')

    @property
    def stream_shapes(self):
        return ((self.obs_dim, self.hidden_dim),
                (self.hidden_dim,),
                (self.hidden_dim, self.hidden_dim),
                (self.hidden_dim,),
                (self.hidden_dim, self.action_dim),
                (self.action_dim,),
                (1, self.action_dim))

    @property
    def stream_sizes(self):
        return [self.obs_dim * self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim * self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim * self.action_dim,
                self.action_dim,
                self.action_dim]

    def scalar_score(self,
                     x: np.ndarray) -> np.ndarray:
        """Calculates a score for the provided tensor x using a ground truth
        oracle function (the goal of the task is to maximize this)

        Args:

        x: np.ndarray
            a batch of sampled designs that will be evaluated by
            an oracle score function

        Returns:

        scores: np.ndarray
            a batch of scores that correspond to the x values provided
            in the function argument
        """

        # extract weights from the vector design
        weights = []
        for s in self.stream_shapes:
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
        obs, done = env.reset(), False
        path_returns = np.zeros([1], dtype=np.float32)
        while not done:
            obs, rew, done, info = env.step(mlp_policy(obs))
            path_returns += rew.astype(np.float32)
        return path_returns
