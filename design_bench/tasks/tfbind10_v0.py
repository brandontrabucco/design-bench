from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
import pandas as pd
import numpy as np
import os


def onehottify(x, n=None, dtype=float):
    """1-hot encode x with the max value n (computed from data if n is None)."""
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]


LETTER_TO_ID = dict(a=0, t=1, c=2, g=3)
ID_TO_LETTER = ['a', 't', 'c', 'g']
INVERSE_LETTER_MAP = dict(a='t', t='a', c='g', g='c')


class TfBind10V0Task(Task):

    def score(self, x):
        return NotImplemented

    def __init__(self,
                 split_percentile=20,
                 transcription_factor='pho4',
                 ys_noise=0.0):
        """Load a fully experimentally characterized dataset of transcription
        factor binding affinity for all possible 10-mers.

        Args:

        split_percentile: int
            the percentile (out of 100) to split the data set by and only
            include samples with score below this percentile
        ys_noise: float
            the number of standard deviations of noise to add to
            the static training dataset y values accompanying this task
        """

        maybe_download('1qeX6vnuOLdj8tzyQ3ub1AjSk4kVlUUQb',
                       os.path.join(DATA_DIR, 'tfbind10_counts.txt'))

        # load experimentally determined binding stability
        data_file = os.path.join(DATA_DIR, 'tfbind10_counts.txt')
        data = pd.read_csv(data_file, sep="\t")
        data = data.loc[data['protein'] == transcription_factor]

        # filter to remove infinite ddG values
        filtered = data.loc[data['ddG'] != np.inf]
        filtered = filtered.loc[data['ddG'] != -np.inf, 'ddG']
        data['ddG'].replace(np.inf, filtered.max(), inplace=True)
        data['ddG'].replace(-np.inf, filtered.min(), inplace=True)

        # load the 10 mer sequences from the dataset
        seq0 = np.char.lower(data["flank"].tolist())
        seq1 = ["".join([INVERSE_LETTER_MAP[c] for c in x]) for x in seq0]

        # load dna 10-mers from the dataset
        x0 = np.array([[LETTER_TO_ID[c] for c in x] for x in seq0])
        x1 = np.array([[LETTER_TO_ID[c] for c in x] for x in seq1])

        # convert the token ids to one-hot representations
        x = np.concatenate([x0, x1], axis=0)
        x = onehottify(x, n=4, dtype=np.float32)

        y0 = data["ddG"].to_numpy()  # "ddG" is the binding stability
        y0 = (y0[:, np.newaxis] - y0.min()) / (y0.max() - y0.min())
        y = np.concatenate([y0, y0], axis=0).astype(np.float32)

        # split the remaining proteins with a threshold
        ind = np.where(y <= np.percentile(y[:, 0], split_percentile))[0]

        # expose the designs
        x = x[ind]
        y = y[ind]

        mean_y = np.mean(y, axis=0, keepdims=True)
        st_y = np.std(y - mean_y, axis=0, keepdims=True)
        y = y + np.random.normal(0.0, 1.0, y.shape) * st_y * ys_noise

        # expose the designs
        self.x = x
        self.y = y
        self.sequences = dict(zip(seq0, y0))
        self.sequences.update(zip(seq1, y0))
        self.score = np.vectorize(self.scalar_score,
                                  signature='(n,4)->(1)')

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

        # lookup the score of a single 8-mer
        word = ''.join(map(lambda token:
                           ID_TO_LETTER[token], np.argmax(x, axis=1)))
        return np.asarray(self.sequences[word], dtype=np.float32)
