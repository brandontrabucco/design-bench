from design_bench.oracles.approximate_oracle import ApproximateOracle
import numpy as np
import abc


class SKLearnOracle(ApproximateOracle, abc.ABC):
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

    fit(np.ndarray, np.ndarray):
        a function that accepts a data set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

    """

    @staticmethod
    def get_subsample_indices(y_dataset, max_samples=5000,
                              min_percentile=0.0, max_percentile=100.0):
        """Helper function for generating indices for subsampling training
        samples from a model-based optimization dataset, particularly when
        using a learned model where not all samples fit into memory

        Arguments:

        y_dataset: np.ndarray
            a numpy array of prediction values from a model-based optimization
            dataset that will be subsampled using the given statistics

        Returns:

        y_dataset: np.ndarray
            a numpy array of smaller size that the original y_dataset, having
            been subsampled using the given statistics

        """

        min_value = np.percentile(y_dataset[:, 0], min_percentile)
        max_value = np.percentile(y_dataset[:, 0], max_percentile)
        indices = np.where(np.logical_and(
            y_dataset[:, 0] >= min_value, y_dataset[:, 0] <= max_value))[0]
        size = indices.size
        return indices[np.random.choice(
            size, size=min(size, max_samples), replace=False)]
