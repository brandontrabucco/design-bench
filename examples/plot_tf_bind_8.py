from design_bench.datasets.discrete import TFBind8Dataset
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn import GaussianProcessOracle
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.oracles.tensorflow import FullyConnectedOracle
from design_bench.oracles.tensorflow import LSTMOracle
from design_bench.oracles.tensorflow import ResNetOracle
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":

    data = pd.read_csv("tf_bind_8_resnet_efficacy.csv")
    data = data.loc[data["Train Split (%)"].isin([20, 40, 60, 80, 100])]
    data["Train Split (%)"] = data["Train Split (%)"].apply(str)
    sns.lineplot(x="Y Percentile", y="Spearman's ρ", hue="Train Split (%)", data=data)
    plt.title("Efficacy of Learned Models")
    plt.savefig("tf_bind_8.png")
