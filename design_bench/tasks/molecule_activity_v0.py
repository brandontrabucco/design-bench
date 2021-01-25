from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle as pkl
import os


def train_one_oracle(target_assay, x_i, y_i):
    """Train a Random Forest Regression Tree using scikit-learn
    and save that classifier to the disk

    Args:

    x: np.ndarray
        the training features for the decision tree represented as a
        matrix with a shape like [n_samples, n_features]
    y: np.ndarray
        the training labels for the decision tree represented as a
        matrix with a shape like [n_samples, 1]
    """

    est = RandomForestRegressor(
        n_estimators=500,
        criterion="mse",
        max_depth=32,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=24,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None)

    est.fit(x_i, y_i[:, 0])
    os.makedirs(os.path.join(
        DATA_DIR, f'molecule_activity_v0'), exist_ok=True)
    with open(os.path.join(
            DATA_DIR, f'molecule_activity_v0/'
                      f'rfr_{target_assay}.pkl'), 'wb') as f:
        pkl.dump(est, f)
    r2 = est.score(x_i, y_i[:, 0])
    print(r2)


class MoleculeActivityV0Task(Task):

    def score(self, x):
        return NotImplemented

    def __init__(self,
                 target_assay=600885,
                 split_percentile=80,
                 ys_noise=0.0):  # this choice has the most spread
        """Create a task for designing super conducting materials that
        have a high critical temperature

        Support target Assay, Data set Size
        688150,               2k
        600886,               5k
        600885,               5k
        688537,               3k
        688597,               3k

        Args:

        split_percentile: int
            the percentile (out of 100) to split the data set by and only
            include samples with score below this percentile
        ys_noise: float
            the number of standard deviations of noise to add to
            the static training dataset y values accompanying this task
        """

        maybe_download('1_8c7uln7vzLbMmoviJhGWRqy-OyMZovc',
                       os.path.join(DATA_DIR, 'molecule_activity_X.npy'))
        maybe_download('1nZgMkKs6_hKb8o1Oy-KhGBLtI7HgqD2K',
                       os.path.join(DATA_DIR, 'molecule_activity_c.npy'))
        maybe_download('1vDycVRmfHi_-NiMjDaFrs-M4r9hZKr8T',
                       os.path.join(DATA_DIR, 'molecule_activity_y.npy'))
        maybe_download('1TFpIiLlpOYUoRRL3vexIhuY2Z_ZuAyxa',
                       os.path.join(DATA_DIR, 'molecule_activity_v0.zip'))

        x = np.load(os.path.join(
            DATA_DIR, 'molecule_activity_X.npy')).astype(np.float32)
        c = np.load(os.path.join(
            DATA_DIR, 'molecule_activity_c.npy')).astype(np.int32)
        y = np.load(os.path.join(
            DATA_DIR, 'molecule_activity_y.npy')).astype(np.float32)

        # select the examples for this assay
        matches = np.where(np.equal(c, target_assay))[0]
        x = x[matches]
        y = y[matches]
        x = np.stack([1.0 - x, x], axis=2)

        # remove all samples above the qth percentile in the data set
        split_temp = np.percentile(y[:, 0], split_percentile)
        indices = np.where(y <= split_temp)[0]
        x = x[indices].astype(np.float32)
        y = y[indices].astype(np.float32)

        mean_y = np.mean(y, axis=0, keepdims=True)
        st_y = np.std(y - mean_y, axis=0, keepdims=True)
        y = y + np.random.normal(0.0, 1.0, y.shape) * st_y * ys_noise
        self.x = x
        self.y = y

        # load the oracle for this assay
        with open(os.path.join(
                DATA_DIR, f'molecule_activity_v0/'
                          f'rfr_{target_assay}.pkl'), 'rb') as f:
            self.oracle = pkl.load(f)
        self.score = np.vectorize(
            self.scalar_score, signature='(n,2)->(1)')

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

        return self.oracle.predict(
            (x[np.newaxis, :, 1] > 0.5).astype(np.float32))
