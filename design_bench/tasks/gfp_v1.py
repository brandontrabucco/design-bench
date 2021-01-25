from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
from tape import TAPETokenizer
from tape import ProteinBertForValuePrediction
import torch
import pandas as pd
import numpy as np
import os


def onehottify(x, n=None, dtype=float):
    """1-hot encode x with the max value n (computed from data if n is None)."""
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]


class GFPV1Task(Task):

    def __init__(self,
                 split_percentile=20,
                 internal_batch_size=1,
                 use_cuda=True,
                 ys_noise=0.0):
        """Load the GFP data set which includes maps from discrete
        protein designs to fluorescence scores

        Args:

        split_percentile: int
            the percentile (out of 100) to split the data set by and only
            include samples with score below this percentile
        ys_noise: float
            the number of standard deviations of noise to add to
            the static training dataset y values accompanying this task
        """

        maybe_download('1_jcPkQ-M1FRhkEONoE57WEbp_Rivkho2',
                       os.path.join(DATA_DIR, 'gfp_data.csv'))
        maybe_download('1R2UaplzHjMaWwsu-bT-kcOAx7GS1HYRu',
                       os.path.join(DATA_DIR, 'gfp_transformer_pretrained.zip'))
        self.batch_size = internal_batch_size
        self.use_cuda = use_cuda

        # load the static dataset
        df = pd.read_csv(os.path.join(DATA_DIR, 'gfp_data.csv'))

        # remove all proteins with fluorescence below the mean
        df = df.loc[df.loc[(df['medianBrightness'] >
                            df['medianBrightness'].mean())].index]

        # remove all proteins with a stop marker
        df = df.loc[df.loc[
            ~df['aaSequence'].str.contains('!')].index]

        # load the tokenizer and pretrained protein model
        self.tokenizer = TAPETokenizer(vocab='iupac')
        self.model = ProteinBertForValuePrediction.from_pretrained(
            os.path.join(DATA_DIR, 'gfp_transformer_pretrained'))
        if self.use_cuda:
            self.model = self.model.cuda()

        # encode the entire dataset using the TAPE tokenizer
        x = np.array([self.tokenizer.encode(x.upper())
                      for x in df['aaSequence']])

        # convert the token ids to one-hot representations
        x = onehottify(x, n=30, dtype=np.float32)

        # format the fluorescence values to a tensor
        y = df['medianBrightness']\
            .to_numpy().astype(np.float32).reshape([-1, 1])

        # split the remaining proteins with a threshold
        ind = np.where(y <= np.percentile(
            y[:, 0], split_percentile))[0]

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

        scores = []
        with torch.no_grad():
            for i in range(x.shape[0] // self.batch_size):

                # run each batch through the torch model
                x_ids = torch.tensor(np.argmax(
                    x[i: (i + 1) * self.batch_size], axis=-1))
                if self.use_cuda:
                    x_ids = x_ids.cuda()
                y = self.model(x_ids)[0].cpu()
                scores.append(y.numpy().reshape([-1, 1]))

            if x.shape[0] % self.batch_size > 0:

                # if there are remaining elements run them at the end
                x_ids = torch.tensor(np.argmax(
                    x[-(x.shape[0] % self.batch_size):], axis=-1))
                if self.use_cuda:
                    x_ids = x_ids.cuda()
                y = self.model(x_ids)[0].cpu()
                scores.append(y.numpy().reshape([-1, 1]))

            # merge together all batches into a single numpy tensor
            return np.concatenate(scores, axis=0)
