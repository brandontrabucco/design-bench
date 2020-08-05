from typing import Tuple
from design_bench.task import Task
import numpy as np
import abc


class ConditionalTask(Task, abc.ABC):
    """A minimalistic task specification that includes designs
    represented by x and scores represented by y

    Attributes:

    x: np.ndarray
        the data set as a numpy tensor whose first 'axis' is
        the batch 'axis' and can take any shape
    c: np.ndarray
        task identifiers as a numpy tensor whose first 'axis' is
        the batch 'axis' and can take any shape
    y: np.ndarray
        the scores for every point in the data set represented
        as a tensor shaped like [num_samples, 1]
    input_shape: Tuple[int]
        Returns the shape of a single sample of x from the data set
        and excludes the batch 'axis'
    input_size: int
        Returns the size of a single sample of x from the data set
        and excludes the batch 'axis'
    condition_shape: Tuple[int]
        Returns the shape of a single sample of c from the data set
        and excludes the batch 'axis'
    condition_size: int
        Returns the size of a single sample of c from the data set
        and excludes the batch 'axis'

    Methods:

    score(self, x: np.ndarray, , c: np.ndarray) -> np.ndarray
        Calculates a score for the provided tensor x using a ground truth
        oracle function (the goal of the task is to maximize this)
    """

    x: np.ndarray = None
    c: np.ndarray = None
    y: np.ndarray = None

    @abc.abstractmethod
    def score(self,
              x: np.ndarray,
              c: np.ndarray) -> np.ndarray:
        """Calculates a score for the provided tensor x using a ground truth
        oracle function (the goal of the task is to maximize this)

        Args:

        x: np.ndarray
            a batch of sampled designs that will be evaluated by
            an oracle score function
        c: np.ndarray
            a batch of task specification variables that determine
            how the designs are scores

        Returns:

        scores: np.ndarray
            a batch of scores that correspond to the x values provided
            in the function argument
        """

        return NotImplemented

    @property
    def input_shape(self) -> Tuple[int]:
        """Returns the shape of a single sample of x from the data set
        and excludes the batch 'axis'

        Returns:

        shape: Tuple[int]
            a tuple of integers that corresponds to the shape of the
            inputs to a function of samples of x
        """

        return self.x.shape[1:]

    @property
    def input_size(self) -> int:
        """Returns the size of a single sample fo x from the data set
        and excludes the batch 'axis'

        Returns:

        size: int
            an integer that represents the total number of channels across
            all axes of a single sample from the data set
        """

        return int(np.prod(self.input_shape))

    @property
    def condition_shape(self) -> Tuple[int]:
        """Returns the shape of a single sample of c from the data set
        and excludes the batch 'axis'

        Returns:

        shape: Tuple[int]
            a tuple of integers that corresponds to the shape of the
            inputs to a function of samples of c
        """

        return self.c.shape[1:]

    @property
    def condition_size(self) -> int:
        """Returns the size of a single sample of c from the data set
        and excludes the batch 'axis'

        Returns:

        size: int
            an integer that represents the total number of channels across
            all axes of a single sample of c
        """

        return int(np.prod(self.condition_shape))
