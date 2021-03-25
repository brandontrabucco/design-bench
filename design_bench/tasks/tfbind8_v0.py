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


class TfBind8V0Task(Task):

    def score(self, x):
        return NotImplemented

    def __init__(self,
                 split_percentile=20,
                 transcription_factor='SIX6_REF_R1',
                 ys_noise=0.0):
        """Load a fully experimentally characterized dataset of transcription
        factor binding affinity for all possible 8-mers.
        Inspired by: https://github.com/samsinai/FLEXS/blob/
        41595eb6901eb2b17d30793c457c107cbb8dc488/
        flexs/landscapes/tf_binding.py

        Args:

        split_percentile: int
            the percentile (out of 100) to split the data set by and only
            include samples with score below this percentile
        ys_noise: float
            the number of standard deviations of noise to add to
            the static training dataset y values accompanying this task
        """

        maybe_download('1xS6N5qSwyFLC-ZPTADYrxZuPHjBkZCrj',
                       os.path.join(DATA_DIR, 'TF_binding_landscapes.zip'))

        # load the static dataset
        tf_dir = os.path.join(os.path.join(
            DATA_DIR, 'TF_binding_landscapes'), 'landscapes')
        data = pd.read_csv(os.path.join(
            tf_dir, f'{transcription_factor}_8mers.txt'), sep="\t")

        # load the 8 mer sequences from the dataset
        seq0 = np.char.lower(data["8-mer"].tolist())
        seq1 = np.char.lower(data["8-mer.1"].tolist())

        # load dna 8-mers from the dataset
        x0 = np.array([[LETTER_TO_ID[c] for c in x.lower()] for x in seq0])
        x1 = np.array([[LETTER_TO_ID[c] for c in x.lower()] for x in seq1])

        # convert the token ids to one-hot representations
        x = np.concatenate([x0, x1], axis=0)
        x = onehottify(x, n=4, dtype=np.float32)

        y0 = data["E-score"].to_numpy()  # "E-score" is enrichment score
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
        self.sequences = dict(zip(seq0, y))
        self.sequences.update(zip(seq1, y))
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
