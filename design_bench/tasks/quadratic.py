from design_bench.task import Task
import numpy as np


class QuadraticTask(Task):

    def __init__(self,
                 global_optimum=(-1.0, 4.0, 2.0, -3.0, 5.0, 1.0,),
                 oracle_noise_std=0.2,
                 dataset_size=100,
                 split_percentile=80,
                 ys_noise=0.0):
        """Create a toy Gaussian Process optimization task using
        the provided sampling bounds

        Args:

        global_optimum: tuple or list
            the maximal location of the quadratic function whose length
            dictates the number of channels in the task dataset
        oracle_noise_std: float
            the standard deviation of the noise added to the function
            used to score the sampled x values
        dataset_size: int
            the number of initial samples to populate the dataset with
            must be a positive integer
        split_percentile: int
            the percentile (out of 100) to split the data set by and only
            include samples with score below this percentile
        ys_noise: float
            the number of standard deviations of noise to add to
            the static training dataset y values accompanying this task
        """

        global_optimum = np.array(global_optimum)[np.newaxis]
        self.global_optimum = global_optimum.astype(np.float32)
        self.oracle_noise_std = oracle_noise_std

        z = np.random.randn(dataset_size, len(global_optimum))
        x = (self.global_optimum + z).astype(np.float32)
        y = self.score(x).astype(np.float32)

        split_value = np.percentile(y[:, 0], split_percentile)
        indices = np.where(y <= split_value)[0]
        y = y[indices]
        x = x[indices]
        self.x = x
        self.y = y

        mean_y = np.mean(self.y, axis=0, keepdims=True)
        st_y = np.std(self.y - mean_y, axis=0, keepdims=True)
        self.y = self.y + np.random.normal(0.0, 1.0, self.y.shape) * st_y * ys_noise

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

        x = x - self.global_optimum
        z = self.oracle_noise_std * np.random.randn(x.shape[0], 1)
        return -np.sum(x ** 2, axis=1, keepdims=True) + z + 6.0
