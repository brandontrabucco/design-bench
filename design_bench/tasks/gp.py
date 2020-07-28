from design_bench.task import Task
import numpy as np


class GP1DTask(Task):

    def __init__(self,
                 dataset_size=100,
                 upper_bound=(4.0,),
                 lower_bound=(-4.0,),
                 noise=0.2):
        """Create a toy Gaussian Process optimization task using
        the provided sampling bounds

        Args:

        dataset_size: int
            the number of initial samples to populate the dataset with
            must be a positive integer
        upper_bound: np.ndarray
            the upper bound on samples drawn from the uniform
            distribution to populate the dataset
        lower_bound: np.ndarray
            the lower bound on samples drawn from the uniform
            distribution to populate the dataset
        noise: float
            the standard deviation of the noise added to the function
            used to score the sampled x values
        """

        self.noise = noise

        x = np.random.uniform(low=np.array(lower_bound)[np.newaxis],
                              high=np.array(upper_bound)[np.newaxis],
                              size=[dataset_size, len(lower_bound)])
        x = x.astype(np.float32)
        self.x = x.reshape([dataset_size, len(lower_bound)])
        self.y = self.score(self.x)

    def score(self,
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

        return (-np.sin(3 * x) - x ** 2 + 0.7 * x +
                self.noise * np.random.randn(*x.shape)).reshape([-1, 1])


class GP2DTask(Task):

    def __init__(self,
                 dataset_size=100,
                 upper_bound=(0.0, 15.0),
                 lower_bound=(-5.0, 10.0)):
        """Create a toy Gaussian Process optimization task using
        the provided sampling bounds

        Args:

        dataset_size: int
            the number of initial samples to populate the dataset with
            must be a positive integer
        upper_bound: np.ndarray
            the upper bound on samples drawn from the uniform
            distribution to populate the dataset
        lower_bound: np.ndarray
            the lower bound on samples drawn from the uniform
            distribution to populate the dataset
        """

        x = np.random.uniform(low=np.array(lower_bound)[np.newaxis],
                              high=np.array(upper_bound)[np.newaxis],
                              size=[dataset_size, len(lower_bound)])
        x = x.astype(np.float32)
        self.x = x.reshape([dataset_size, len(lower_bound)])
        self.y = self.score(self.x)

    def score(self,
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

        a = 1
        b = 5.1 / (4 * np.pi * np.pi)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1.0 / (8 * np.pi)
        return (a * ((x[:, 1] - b * (x[:, 0] ** 2) + c * x[:, 0] - r) ** 2) +
                s * (1 - t) * np.cos(x[:, 0]) + s).reshape([-1, 1])
