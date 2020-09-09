from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
from morphing_agents.mujoco.ant.env import MorphingAntEnv
from morphing_agents.mujoco.ant.elements import LEG
from morphing_agents.mujoco.ant.elements import LEG_LOWER_BOUND
from morphing_agents.mujoco.ant.elements import LEG_UPPER_BOUND
import numpy as np
import os
import pickle as pkl


class MorphologyV0Task(Task):

    def score(self, x):
        return NotImplemented

    def __init__(self,
                 env_class=MorphingAntEnv,
                 elements=4,
                 env_element=LEG,
                 env_element_lb=LEG_LOWER_BOUND,
                 env_element_ub=LEG_UPPER_BOUND,
                 oracle_weights='ant_oracle.pkl',
                 x_file='ant_morphology_X.npy',
                 y_file='ant_morphology_y.npy',
                 split_percentile=60,
                 num_rollouts=16,
                 rollout_horizon=100):
        """Load static datasets of weights and their corresponding
        expected returns from the disk

        Args:

        num_parallel: int
            the number of parallel trials in the docker container
        num_gpus: int
            the number of gpus to use in this docker container
        n_envs: int
            the number of parallel sampling environments for PPO
        max_episode_steps: int
            the maximum length of a episode when collecting samples
        total_timesteps: int
            the maximum number of samples to collect from the environment
        domain: str
            the particular morphology domain such as 'ant' or 'dog'
        """

        self.score = np.vectorize(self.scalar_score, signature='(n)->(1)')
        maybe_download('12H-4AvzpMVmq7M7b7nD_RPu5GNeCuCbu',
                       os.path.join(DATA_DIR, 'ant_morphology_X.npy'))
        maybe_download('1uSF6oc7OlLGioe_sZQwmjibuRbPCYRGu',
                       os.path.join(DATA_DIR, 'ant_morphology_y.npy'))
        maybe_download('1FJY6LQG3kvLIx00WT2XVGyfU9JrmO0T0',
                       os.path.join(DATA_DIR, 'ant_oracle.pkl'))
        maybe_download('17PqWmmbIiUwNBxLwhjTnV38Dx2ejefCQ',
                       os.path.join(DATA_DIR, 'dkitty_morphology_X.npy'))
        maybe_download('1DggTSnRn_SzmYonbLgLppzWC8OtqYAb8',
                       os.path.join(DATA_DIR, 'dkitty_morphology_y.npy'))
        maybe_download('1GpZc9UezjyvAn_ejwG3cibq1stQMjiiM',
                       os.path.join(DATA_DIR, 'dkitty_oracle.pkl'))

        self.env_class = env_class
        self.elements = elements
        self.env_element = env_element
        self.lb = env_element_lb
        self.ub = env_element_ub
        self.num_rollouts = num_rollouts
        self.rollout_horizon = rollout_horizon
        with open(os.path.join(
                DATA_DIR, oracle_weights), 'rb') as f:
            self.weights = pkl.load(f)

        x = np.load(os.path.join(DATA_DIR, x_file))
        y = np.load(os.path.join(DATA_DIR, y_file))
        x = x.astype(np.float32)
        y = y.astype(np.float32).reshape([-1, 1])

        # remove all samples above the qth percentile in the data set
        split_temp = np.percentile(y[:, 0], split_percentile)
        indices = np.where(y < split_temp)[0]
        x = x[indices].astype(np.float32)
        y = y[indices].astype(np.float32)

        # normalize the x values
        self.m = np.mean(x, axis=0, keepdims=True)
        self.st = np.std(x - self.m, axis=0, keepdims=True)
        self.st = np.where(np.equal(self.st, 0), 1.0, self.st)
        self.y = y
        self.x = (x - self.m) / self.st

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

        # create a policy forward pass in numpy
        def mlp_policy(h):
            h = np.maximum(0.0, h @ self.weights[0] + self.weights[1])
            h = np.maximum(0.0, h @ self.weights[2] + self.weights[3])
            h = h @ self.weights[4] + self.weights[5]
            return np.tanh(np.split(h, 2)[0])

        # convert vectors to morphologies
        x = x * self.st[0] + self.m[0]
        env = self.env_class(expose_design=False, fixed_design=[
            self.env_element(*np.clip(np.array(xi), self.lb, self.ub))
            for xi in np.split(x, self.elements)])

        # do many rollouts using a pretrained agent
        average_returns = []
        for i in range(self.num_rollouts):
            obs = env.reset()
            average_returns.append(np.zeros([], dtype=np.float32))
            for t in range(self.rollout_horizon):
                obs, rew, done, info = env.step(mlp_policy(obs))
                average_returns[-1] += rew.astype(np.float32)
                if done:
                    break

        # we average here so as to reduce randomness
        return np.mean(average_returns).reshape([1])
