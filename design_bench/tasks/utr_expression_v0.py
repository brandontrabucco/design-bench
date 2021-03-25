from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
import pandas as pd
import numpy as np
import os
import keras


def onehottify(x, n=None, dtype=float):
    """1-hot encode x with the max value n (computed from data if n is None)."""
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]


LETTER_TO_ID = dict(a=0, c=1, g=2, t=3)
ID_TO_LETTER = ['a', 'c', 'g', 't']


class UTRExpressionV0Task(Task):

    def __init__(self,
                 split_percentile=20,
                 ys_noise=0.0):
        """Load a dataset of DNA sequences correspond to 5'UTR sequences
        and their corresponding gene expression levels
        Inspired by: https://github.com/pjsample/human_5utr_modeling

        Args:

        split_percentile: int
            the percentile (out of 100) to split the data set by and only
            include samples with score below this percentile
        ys_noise: float
            the number of standard deviations of noise to add to
            the static training dataset y values accompanying this task
        """

        maybe_download('1pRypiGVYl-kmJZaMhVbuA1PEvqauWBBM',
                       os.path.join(DATA_DIR, 'utr.zip'))
        utr_dir = os.path.join(DATA_DIR, 'utr')

        # load the static dataset
        df = pd.read_csv(os.path.join(utr_dir, 'egfp_unmod_1.csv'))
        df.sort_values('total_reads', inplace=True, ascending=False)
        df.reset_index(inplace=True, drop=True)
        df = df.iloc[:280000]

        # load the 8 mer sequences from the dataset
        seq = np.char.lower(df["utr"].tolist())

        # load dna 8-mers from the dataset
        x = np.array([[LETTER_TO_ID[c] for c in x.lower()] for x in seq])

        # convert the token ids to one-hot representations
        x = onehottify(x, n=4, dtype=np.float32)

        self.model = keras.models.load_model(
            os.path.join(utr_dir, 'main_MRL_model.hdf5'))
        y = self.model.predict(x)  # label the data set using the trained model

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

        # use the trained model to predict the score
        return self.model.predict(x)
