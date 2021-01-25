"""Code is primarily adapted from:
https://github.com/dhbrookes/CbAS/blob/master/util.py
"""


from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
import pandas as pd
import numpy as np
import os


class GFPV0Task(Task):

    def __init__(self,
                 seed=0,
                 split_percentile=100,
                 ys_noise=0.0):
        """Load the GFP data set which includes maps from discrete
        protein designs to fluorescence scores

        Args:

        seed: int
            the random seed used for this experiment that determines the
            sampled values of data set noise
        split_percentile: int
            the percentile (out of 100) to split the data set by and only
            include samples with score below this percentile
        ys_noise: float
            the number of standard deviations of noise to add to
            the static training dataset y values accompanying this task
        """

        maybe_download('1UO8L3uOp141m2v5dVlpGZ4tZ42XIJ4Vq',
                       os.path.join(DATA_DIR, 'gfp_gt_evals.npy'))
        maybe_download('1DeOoYQs5GEis3jIYsbGxuemjtsBiUSJm',
                       os.path.join(DATA_DIR, 'gfp_gpy.npy'))
        maybe_download('10xMOWXZjGOKLokO4jP6ya29-ZD2tb46X',
                       os.path.join(DATA_DIR, 'gfp_gpX.npy'))
        maybe_download('18EvOK25vmPvRGNbviv1Oep2CPXt3UrLt',
                       os.path.join(DATA_DIR, 'gfp_gpparams.npy'))
        maybe_download('1ySC8Rkfye6JfRKqoDS_KAXqUQTKtrbvZ',
                       os.path.join(DATA_DIR, 'gfp_gpKinv.npy'))
        maybe_download('1tRvY0W4ygoPxytdhAWZuwSQmvNj2QEtK',
                       os.path.join(DATA_DIR, 'gfp_gpK.npy'))
        maybe_download('1_jcPkQ-M1FRhkEONoE57WEbp_Rivkho2',
                       os.path.join(DATA_DIR, 'gfp_data.csv'))

        # load the static dataset
        self.sequence_gp = SequenceGP(load=True)
        x, _, y = get_experimental_X_y(random_state=seed)

        # cast everything to floats
        x = x.astype(np.float32)
        y = y.astype(np.float32).reshape([-1, 1])

        split_value = np.percentile(y[:, 0], split_percentile)
        indices = np.where(y <= split_value)[0]
        y = y[indices]
        x = x[indices]

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

        return self.sequence_gp.predict(np.argmax(
            x, axis=-1)).astype(np.float32).reshape([-1, 1])


class SequenceGP(object):

    def __init__(self,
                 load=False,
                 X_train=None,
                 y_train=None,
                 length_scale=1,
                 homo_noise=0.1,
                 load_prefix="gfp_gp",
                 k_beta=0.1,
                 c=1,
                 d=2):
        if load:
            self.load(prefix=load_prefix)
        else:
            assert X_train is not None and y_train is not None
            self.X_ = np.copy(X_train)
            self.y_ = np.copy(y_train).reshape((y_train.shape[0], 1))
            self.N_ = self.X_.shape[0]
            self.params_ = np.array([homo_noise, k_beta, c, d])
            self.K_ = None
            self.Kinv_ = None

    def _kernel(self,
                Xi,
                Xj):
        beta = self.params_[1]
        c = self.params_[2]
        d = self.params_[3]
        kij = np.prod(BLOSUM[(Xi, Xj)] ** beta)
        kii = np.prod(BLOSUM[(Xi, Xi)] ** beta)
        kjj = np.prod(BLOSUM[(Xj, Xj)] ** beta)
        k = kij / (np.sqrt(kii * kjj))
        k = np.exp(c * k)
        #         k = (k+c)**d
        return k

    def _fill_K(self,
                print_every=100):
        self.K_ = np.zeros((self.N_, self.N_))
        total = self.N_ * (self.N_ + 1) / 2
        m = 0
        homo_noise = self.params_[0]
        for i in range(self.N_):
            for j in range(i, self.N_):
                kij = self._kernel(self.X_[i], self.X_[j])
                if i == j:
                    kij += homo_noise
                self.K_[i, j] = kij
                self.K_[j, i] = kij

                m += 1
                if m % print_every == 0:
                    print("Number of K elements filled: %i / %i" % (m, total))

    def _invert_K(self):
        print("Inverting K...")
        self.Kinv_ = np.linalg.inv(self.K_)
        print("Done inverting K.")

    def build(self,
              print_every=100):
        self._fill_K(print_every=print_every)
        self._invert_K()

    def predict(self,
                Xstar,
                print_every=None,
                predict_variance=False):
        M = len(Xstar)
        Kstar = np.zeros((M, self.N_))
        total = M * self.N_
        m = 0
        for i in range(M):
            for j in range(self.N_):
                kij = self._kernel(Xstar[i], self.X_[j])
                Kstar[i, j] = kij
                m += 1
                if print_every is not None:
                    if m % print_every == 0:
                        print("Number of Kstar elements filled: %i / %i" % (m, total))
        mu_star = np.matmul(Kstar, np.matmul(self.Kinv_, self.y_))
        return mu_star

    def save(self,
             prefix="gfp_gp"):
        np.save(os.path.join(DATA_DIR, prefix + "X.npy", self.X_))
        np.save(os.path.join(DATA_DIR, prefix + "y.npy", self.y_))
        np.save(os.path.join(DATA_DIR, prefix + "K.npy", self.K_))
        np.save(os.path.join(DATA_DIR, prefix + "Kinv.npy", self.Kinv_))
        np.save(os.path.join(DATA_DIR, prefix + "params.npy", self.params_))

    def load(self,
             prefix="gfp_gp"):
        self.X_ = np.load(os.path.join(DATA_DIR, prefix + "X.npy"))
        self.y_ = np.load(os.path.join(DATA_DIR, prefix + "y.npy"))
        self.K_ = np.load(os.path.join(DATA_DIR, prefix + "K.npy"))
        self.Kinv_ = np.load(os.path.join(DATA_DIR, prefix + "Kinv.npy"))
        self.params_ = np.load(os.path.join(DATA_DIR, prefix + "params.npy"))
        self.N_ = self.X_.shape[0]


AA = ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i',
      'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v']


AA_IDX = {AA[i]: i for i in range(len(AA))}


BLOSUM = np.array([
[3.9029,0.6127,0.5883,0.5446,0.8680,0.7568,0.7413,1.0569,0.5694,0.6325,0.6019,0.7754,0.7232,0.4649,0.7541,1.4721,0.9844,0.4165,0.5426,0.9365],
[0.6127,6.6656,0.8586,0.5732,0.3089,1.4058,0.9608,0.4500,0.9170,0.3548,0.4739,2.0768,0.6226,0.3807,0.4815,0.7672,0.6778,0.3951,0.5560,0.4201],
[0.5883,0.8586,7.0941,1.5539,0.3978,1.0006,0.9113,0.8637,1.2220,0.3279,0.3100,0.9398,0.4745,0.3543,0.4999,1.2315,0.9842,0.2778,0.4860,0.3690],
[0.5446,0.5732,1.5539,7.3979,0.3015,0.8971,1.6878,0.6343,0.6786,0.3390,0.2866,0.7841,0.3465,0.2990,0.5987,0.9135,0.6948,0.2321,0.3457,0.3365],
[0.8680,0.3089,0.3978,0.3015,19.5766,0.3658,0.2859,0.4204,0.3550,0.6535,0.6423,0.3491,0.6114,0.4390,0.3796,0.7384,0.7406,0.4500,0.4342,0.7558],
[0.7568,1.4058,1.0006,0.8971,0.3658,6.2444,1.9017,0.5386,1.1680,0.3829,0.4773,1.5543,0.8643,0.3340,0.6413,0.9656,0.7913,0.5094,0.6111,0.4668],
[0.7413,0.9608,0.9113,1.6878,0.2859,1.9017,5.4695,0.4813,0.9600,0.3305,0.3729,1.3083,0.5003,0.3307,0.6792,0.9504,0.7414,0.3743,0.4965,0.4289],
[1.0569,0.4500,0.8637,0.6343,0.4204,0.5386,0.4813,6.8763,0.4930,0.2750,0.2845,0.5889,0.3955,0.3406,0.4774,0.9036,0.5793,0.4217,0.3487,0.3370],
[0.5694,0.9170,1.2220,0.6786,0.3550,1.1680,0.9600,0.4930,13.5060,0.3263,0.3807,0.7789,0.5841,0.6520,0.4729,0.7367,0.5575,0.4441,1.7979,0.3394],
[0.6325,0.3548,0.3279,0.3390,0.6535,0.3829,0.3305,0.2750,0.3263,3.9979,1.6944,0.3964,1.4777,0.9458,0.3847,0.4432,0.7798,0.4089,0.6304,2.4175],
[0.6019,0.4739,0.3100,0.2866,0.6423,0.4773,0.3729,0.2845,0.3807,1.6944,3.7966,0.4283,1.9943,1.1546,0.3711,0.4289,0.6603,0.5680,0.6921,1.3142],
[0.7754,2.0768,0.9398,0.7841,0.3491,1.5543,1.3083,0.5889,0.7789,0.3964,0.4283,4.7643,0.6253,0.3440,0.7038,0.9319,0.7929,0.3589,0.5322,0.4565],
[0.7232,0.6226,0.4745,0.3465,0.6114,0.8643,0.5003,0.3955,0.5841,1.4777,1.9943,0.6253,6.4815,1.0044,0.4239,0.5986,0.7938,0.6103,0.7084,1.2689],
[0.4649,0.3807,0.3543,0.2990,0.4390,0.3340,0.3307,0.3406,0.6520,0.9458,1.1546,0.3440,1.0044,8.1288,0.2874,0.4400,0.4817,1.3744,2.7694,0.7451],
[0.7541,0.4815,0.4999,0.5987,0.3796,0.6413,0.6792,0.4774,0.4729,0.3847,0.3711,0.7038,0.4239,0.2874,12.8375,0.7555,0.6889,0.2818,0.3635,0.4431],
[1.4721,0.7672,1.2315,0.9135,0.7384,0.9656,0.9504,0.9036,0.7367,0.4432,0.4289,0.9319,0.5986,0.4400,0.7555,3.8428,1.6139,0.3853,0.5575,0.5652],
[0.9844,0.6778,0.9842,0.6948,0.7406,0.7913,0.7414,0.5793,0.5575,0.7798,0.6603,0.7929,0.7938,0.4817,0.6889,1.6139,4.8321,0.4309,0.5732,0.9809],
[0.4165,0.3951,0.2778,0.2321,0.4500,0.5094,0.3743,0.4217,0.4441,0.4089,0.5680,0.3589,0.6103,1.3744,0.2818,0.3853,0.4309,38.1078,2.1098,0.3745],
[0.5426,0.5560,0.4860,0.3457,0.4342,0.6111,0.4965,0.3487,1.7979,0.6304,0.6921,0.5322,0.7084,2.7694,0.3635,0.5575,0.5732,2.1098,9.8322,0.6580],
[0.9365,0.4201,0.3690,0.3365,0.7558,0.4668,0.4289,0.3370,0.3394,2.4175,1.3142,0.4565,1.2689,0.7451,0.4431,0.5652,0.9809,0.3745,0.6580,3.6922]]
)


def one_hot_encode_aa(aa_str,
                      pad=None):
    M = len(aa_str)
    aa_arr = np.zeros((M, 20), dtype=int)
    for i in range(M):
        aa_arr[i, AA_IDX[aa_str[i]]] = 1
    return aa_arr


def partition_data(X,
                   y,
                   percentile=40,
                   train_size=1000,
                   random_state=1,
                   return_test=False):
    np.random.seed(random_state)
    assert (percentile * 0.01 * len(y) >= train_size)
    y_percentile = np.percentile(y, percentile)
    idx = np.where(y < y_percentile)[0]
    #     print(y_percentile)
    rand_idx = np.random.choice(idx, size=train_size, replace=False)
    X_train = X[rand_idx]
    y_train = y[rand_idx]
    if return_test:
        test_idx = [i for i in idx if i not in rand_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train


def get_experimental_X_y(random_state=1,
                         train_size=5000,
                         return_test=False,
                         return_all=False):
    df = pd.read_csv(os.path.join(DATA_DIR, 'gfp_data.csv'))
    X, _ = get_gfp_X_y_aa(df, large_only=True, ignore_stops=True)
    y_gt = np.load(os.path.join(DATA_DIR, "gfp_gt_evals.npy"))
    if return_test:
        X_train, gt_train, X_test, gt_test = partition_data(
            X, y_gt, percentile=20, train_size=train_size,
            random_state=random_state, return_test=return_test)
        np.random.seed(random_state)
        gt_var = 0.01
        y_train = gt_train + np.random.randn(*gt_train.shape) * gt_var
        y_test = gt_test + np.random.randn(*gt_test.shape) * gt_var
        return X_train, y_train, gt_train, X_test, y_test, gt_test
    else:
        X_train, gt_train = partition_data(
            X, y_gt, percentile=20, train_size=train_size,
            random_state=random_state, return_test=return_test)
        np.random.seed(random_state)
        gt_var = 0.01
        y_train = gt_train + np.random.randn(*gt_train.shape) * gt_var
        return X_train, y_train, gt_train


def get_gfp_X_y_aa(data_df,
                   large_only=False,
                   ignore_stops=True,
                   return_str=False):
    if large_only:
        idx = data_df.loc[(data_df['medianBrightness'] >
                           data_df['medianBrightness'].mean())].index
    else:
        idx = data_df.index
    data_df = data_df.loc[idx]

    if ignore_stops:
        idx = data_df.loc[~data_df['aaSequence'].str.contains('!')].index
    data_df = data_df.loc[idx]
    seqs = data_df['aaSequence']

    M = len(seqs[0])
    N = len(seqs)
    X = np.zeros((N, M, 20))
    j = 0
    for i in idx:
        X[j] = one_hot_encode_aa(seqs[i])
        j += 1
    y_raw = np.array(data_df['medianBrightness'][idx])
    y = y_raw
    if return_str:
        return X, y, list(data_df['aaSequence'])
    else:
        return X, y
