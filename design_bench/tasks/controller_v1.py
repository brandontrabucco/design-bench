from design_bench import DATA_DIR
from design_bench.task import Task
import numpy as np
import pickle as pkl
import gym
import os


class ControllerV1Task(Task):

    def score(self, x):
        return NotImplemented

    def __init__(self,
                 obs_dim=11,
                 action_dim=3,
                 hidden_dim=64,
                 env_name='Hopper-v2',
                 x_file='hopper_controller_v1_X.pkl',
                 y_file='hopper_controller_v1_y.pkl'):
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
        """

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.env_name = env_name

        x = np.load(os.path.join(DATA_DIR, x_file))
        y = np.load(os.path.join(DATA_DIR, y_file))
        x = x.astype(np.float32)
        y = y.astype(np.float32).reshape([-1, 1])

        self.x = x
        self.y = y
        self.score = np.vectorize(self.scalar_score,
                                  signature='(n)->(1)')


    @property
    def stream_shapes(self):
        return ((self.hidden_dim, self.obs_dim),
                (self.hidden_dim,),
                (self.action_dim, self.hidden_dim),
                (self.action_dim,),
                (1,),
                (1,))

    @property
    def stream_sizes(self):
        return [self.obs_dim * self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim * self.action_dim,
                self.action_dim,
                1,
                1]

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

        # the final two weights are for log_std and not used
        weights.pop(-1)
        weights.pop(-1)

        # create a policy forward pass in numpy
        def mlp_policy(h):
            h = np.maximum(0, h @ weights[0].T + weights[1])
            return h @ weights[2].T + weights[3]

        # make a copy of the policy and the environment
        env = gym.make(self.env_name)

        # perform a single rollout for quick evaluation
        obs, done = env.reset(), False
        path_returns = np.zeros([1], dtype=np.float32)
        while not done:
            obs, rew, done, info = env.step(mlp_policy(obs))
            path_returns += rew.astype(np.float32)
        return path_returns