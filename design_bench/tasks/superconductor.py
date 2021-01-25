from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import pickle as pkl
import os


def train_oracle(x, y):
    """Train a Gradient Boosted Regression Tree using scikit-learn
    and save that classifier to the disk

    Args:

    x: np.ndarray
        the training features for the decision tree represented as a
        matrix with a shape like [n_samples, n_features]
    y: np.ndarray
        the training labels for the decision tree represented as a
        matrix with a shape like [n_samples, 1]
    """

    est = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.02,
        n_estimators=374,
        subsample=0.50,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=16,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        init=None,
        random_state=None,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        presort='deprecated',
        validation_fraction=1 / 3,
        n_iter_no_change=None,
        tol=0.0001,
        ccp_alpha=0.0)

    est.fit(x, y[:, 0])
    with open(os.path.join(
            DATA_DIR, 'superconductor_oracle.pkl'), 'wb') as f:
        pkl.dump(est, f)


class SuperconductorTask(Task):

    def __init__(self,
                 split_percentile=80,
                 ys_noise=0.0):
        """Create a task for designing super conducting materials that
        have a high critical temperature

        Args:

        split_percentile: int
            the percentile (out of 100) to split the data set by and only
            include samples with score below this percentile
        ys_noise: float
            the number of standard deviations of noise to add to
            the static training dataset y values accompanying this task
        """

        maybe_download('1AguXqbNrSc665sablzVJh4RHLodeXglx',
                       os.path.join(DATA_DIR, 'superconductor_unique_m.csv'))
        maybe_download('15luLFnXpKDBi1jPL-NJlfeIGNI1QyZsf',
                       os.path.join(DATA_DIR, 'superconductor_train.csv'))
        maybe_download('1GvpMGXNuGVIoNgd0o7r-pXBQa1Zb-NSX',
                       os.path.join(DATA_DIR, 'superconductor_oracle.pkl'))

        train = pd.read_csv(os.path.join(DATA_DIR, 'superconductor_train.csv'))
        data = train.to_numpy()
        y = data[:, -1:]
        x = data[:, :-1]

        split_value = np.percentile(y[:, 0], split_percentile)
        indices = np.where(y <= split_value)[0]
        y = y[indices]
        x = x[indices]

        self.m = np.mean(x, axis=0, keepdims=True)
        self.st = np.std(x - self.m, axis=0, keepdims=True)

        mean_y = np.mean(y, axis=0, keepdims=True)
        st_y = np.std(y - mean_y, axis=0, keepdims=True)
        y = y + np.random.normal(0.0, 1.0, y.shape) * st_y * ys_noise

        self.y = y
        self.x = (x - self.m) / self.st

        with open(os.path.join(
                DATA_DIR, 'superconductor_oracle.pkl'), 'rb') as f:
            self.est = pkl.load(f)

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

        return self.est.predict(x * self.st + self.m).reshape([-1, 1])
