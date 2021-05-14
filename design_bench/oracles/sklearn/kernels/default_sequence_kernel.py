from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import GenericKernelMixin
import numpy as np


class DefaultSequenceKernel(GenericKernelMixin, Kernel):
    """Kernel for the sklearn GaussianProcessRegressor on discrete data
    when a domain-specific kernel is not already known

    Initialize with:

    >>> from design_bench.oracles.sklearn.kernels import DefaultSequenceKernel
    >>> from design_bench.oracles.sklearn import GaussianProcessOracle
    >>> from design_bench.datasets.discrete.gfp_dataset import GFPDataset
    >>> dataset = GFPDataset()
    >>> kernel = DefaultSequenceKernel(dataset.num_classes)
    >>> gp = GaussianProcessOracle(dataset, kernel=kernel)

    """

    def __init__(self, size=None, diagonal=1.0, off_diagonal=0.1):
        self.size = size
        self.diagonal = diagonal
        self.off_diagonal = off_diagonal
        self.kernel_matrix = np.full((size, size), off_diagonal)
        np.fill_diagonal(self.kernel_matrix, diagonal)

    def evaluate_kernel(self, x, y):
        return self.kernel_matrix[x][:, y].sum()

    def __call__(self, X, Y=None, eval_gradient=False):
        return np.array([[self.evaluate_kernel(
            x, y) for y in (X if Y is None else Y)] for x in X])

    def diag(self, X):
        return np.array([self.evaluate_kernel(x, x) for x in X])

    def is_stationary(self):
        return False  # the kernel is fixed in advance
