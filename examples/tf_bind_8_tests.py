from design_bench.datasets.discrete import TFBind8Dataset
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn import GaussianProcessOracle
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.oracles.tensorflow import FullyConnectedOracle
from design_bench.oracles.tensorflow import LSTMOracle
from design_bench.oracles.tensorflow import ResNetOracle
from design_bench.oracles.tensorflow import TransformerOracle
from scipy import stats
import numpy as np
import pandas as pd


if __name__ == "__main__":

    dataset = TFBind8Dataset()
    percentiles = list(range(10, 101, 5))
    data = pd.DataFrame(columns=["Train Split (%)",
                                 "Y Percentile",
                                 "Spearman's ρ"])

    for p in percentiles:

        dataset.subsample(min_percentile=0, max_percentile=p)
        oracle = LSTMOracle(dataset, fit=True,
                            learning_rate=0.001, epochs=50)

        dataset.subsample(min_percentile=0, max_percentile=100)
        ground_truth_ys = []
        oracle_ys = []

        for x, y in dataset.iterate_batches(oracle.internal_batch_size):
            ground_truth_ys.append(y)
            oracle_ys.append(oracle.predict(x))

        ground_truth_ys = np.concatenate(ground_truth_ys, axis=0)[:, 0]
        oracle_ys = np.concatenate(oracle_ys, axis=0)[:, 0]

        for p_lower, p_upper in zip(
                percentiles[:-1], percentiles[1:]):

            y_lower = np.percentile(ground_truth_ys, p_lower)
            y_upper = np.percentile(ground_truth_ys, p_upper)

            indices = np.where(np.logical_and(
                ground_truth_ys >= y_lower,
                ground_truth_ys <= y_upper))[0]

            rank_correlation = stats.spearmanr(
                ground_truth_ys[indices], oracle_ys[indices])[0]

            data = data.append({
                "Train Split (%)": p,
                "Y Percentile": p_upper,
                "Spearman's ρ": rank_correlation},
                ignore_index=True)
            data.to_csv("tf_bind_8_resnet_efficacy.csv")
