from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.conditional_task import ConditionalTask
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle as pkl
import os


def train_one_oracle(unique_c_i, x_i, y_i):
    """Train a Random Forest Regression Tree using scikit-learn
    and save that classifier to the disk

    Args:

    x: np.ndarray
        the training features for the decision tree represented as a
        matrix with a shape like [n_samples, n_features]
    c: np.ndarray
        an additional matrix of discrete ids that conditions the values y
        a matrix shaped like [n_samples]
    y: np.ndarray
        the training labels for the decision tree represented as a
        matrix with a shape like [n_samples, 1]
    """

    est = RandomForestRegressor(
        n_estimators=200,
        criterion="mse",
        max_depth=16,
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

    val_size = x_i.shape[0] // 4
    indices = np.arange(x_i.shape[0])
    np.random.shuffle(indices)
    x_train = x_i[indices[val_size:]]
    y_train = y_i[indices[val_size:]]
    x_test = x_i[indices[:val_size]]
    y_test = y_i[indices[:val_size]]

    est.fit(x_train, y_train[:, 0])
    with open(os.path.join(
            DATA_DIR, f'molecule_activity/rfr'
                      f'_{unique_c_i}.pkl'), 'wb') as f:
        pkl.dump(est, f)
    return est.score(x_test, y_test[:, 0])


def train_oracle(x, c, y):
    """Train a Random Forest Regression Tree using scikit-learn
    and save that classifier to the disk

    Args:

    x: np.ndarray
        the training features for the decision tree represented as a
        matrix with a shape like [n_samples, n_features]
    c: np.ndarray
        an additional matrix of discrete ids that conditions the values y
        a matrix shaped like [n_samples]
    y: np.ndarray
        the training labels for the decision tree represented as a
        matrix with a shape like [n_samples, 1]
    """

    r2 = []
    for unique_c_i in np.unique(c):
        indices = np.where(np.equal(c, unique_c_i))[0]
        r2.append(train_one_oracle(unique_c_i, x[indices], y[indices]))
    np.save(os.path.join(DATA_DIR, 'molecule_activity_rfr_r2.npy'), r2)


class MoleculeActivityTask(ConditionalTask):

    def score(self, x, c):
        return NotImplemented

    def __init__(self,
                 min_validation_r2=0.0,
                 min_num_samples=100):
        """Create a task for designing super conducting materials that
        have a high critical temperature
        """

        maybe_download('1_8c7uln7vzLbMmoviJhGWRqy-OyMZovc',
                       os.path.join(DATA_DIR, 'molecule_activity_X.npy'))
        maybe_download('1nZgMkKs6_hKb8o1Oy-KhGBLtI7HgqD2K',
                       os.path.join(DATA_DIR, 'molecule_activity_c.npy'))
        maybe_download('1vDycVRmfHi_-NiMjDaFrs-M4r9hZKr8T',
                       os.path.join(DATA_DIR, 'molecule_activity_y.npy'))
        maybe_download('1LAxrxhcsRhjE_p4J-Ft6TGd-AxeXGgyq',
                       os.path.join(DATA_DIR, 'molecule_activity_rfr_r2.npy'))
        maybe_download('17qbXsWQyaQ1jIQeOkMLC7GX6McIpgYGh',
                       os.path.join(DATA_DIR, 'molecule_activity.zip'))

        x = np.load(os.path.join(
            DATA_DIR, 'molecule_activity_X.npy')).astype(np.float32)
        c = np.load(os.path.join(
            DATA_DIR, 'molecule_activity_c.npy')).astype(np.int32)
        y = np.load(os.path.join(
            DATA_DIR, 'molecule_activity_y.npy')).astype(np.float32)

        r2 = np.load(os.path.join(
            DATA_DIR, 'molecule_activity_rfr_r2.npy')).astype(np.float32)
        unique, counts = np.unique(c, return_counts=True)
        indices = np.where(np.logical_and(
            counts > min_num_samples, r2 > min_validation_r2))[0]

        self.x = []
        self.c = []
        self.y = []
        self.assays = []
        self.counts = []

        for idx in indices:
            unique_assay = unique[idx]
            matches = np.where(np.equal(c, unique_assay))[0]
            self.x.append(x[matches])
            self.c.append(c[matches])
            self.y.append(y[matches])
            self.assays.append([unique_assay])
            self.counts.append([counts[idx]])

        self.x = np.concatenate(self.x, axis=0)
        self.c = np.concatenate(self.c, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        self.assays = np.concatenate(self.assays, axis=0)
        self.counts = np.concatenate(self.counts, axis=0)
        self.x = np.stack([1.0 - self.x, self.x], axis=2)

        self.oracles = []
        for assay_id in self.assays:
            with open(os.path.join(
                    DATA_DIR, f'molecule_activity/rfr'
                              f'_{assay_id}.pkl'), 'rb') as f:
                self.oracles.append(pkl.load(f))
        self.score = np.vectorize(
            self.scalar_score, signature='(n,2),()->(1)')

    def scalar_score(self,
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

        est = self.oracles[int(np.where(np.equal(self.assays, c))[0])]
        return est.predict((x[np.newaxis, :, 1] > 0.5).astype(np.float32))
