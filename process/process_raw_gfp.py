from design_bench import DATA_DIR
from design_bench import maybe_download
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import argparse
import os
import math


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw GFP")
    parser.add_argument("--shard-folder", type=str, default="./gfp")
    parser.add_argument("--samples-per-shard", type=int, default=5000)
    args = parser.parse_args()

    # download the gfp dataset if not already
    maybe_download('1_jcPkQ-M1FRhkEONoE57WEbp_Rivkho2',
                   os.path.join(DATA_DIR, 'gfp_data.csv'))

    # load the static dataset
    df = pd.read_csv(os.path.join(DATA_DIR, 'gfp_data.csv'))

    # remove all proteins with a stop marker
    df = df.loc[df.loc[
        ~df['aaSequence'].str.contains('!')].index]

    # extract the amino acid sequences for each protein
    x = np.array([list(x) for x in
                  df['aaSequence'].to_list()])

    # build an integer encoder for the allowed amino acids
    encoder = OrdinalEncoder(dtype=np.int32)
    encoder.fit(x.reshape((-1, 1)))

    # encode a dataset of amino acid sequences into categorical features
    x = encoder.transform(x.reshape((-1, 1))).reshape(x.shape)

    # format the fluorescence values to a tensor
    y = df['medianBrightness'] \
        .to_numpy().astype(np.float32).reshape([-1, 1])

    # calculate the number of batches per single shard
    batch_per_shard = int(math.ceil(
        y.shape[0] / args.samples_per_shard))

    # loop once per batch contained in the shard

    os.makedirs(args.shard_folder, exist_ok=True)
    for shard_id in range(batch_per_shard):

        # slice out a component of the current shard
        x_sliced = x[shard_id * args.samples_per_shard:
                     (shard_id + 1) * args.samples_per_shard]
        y_sliced = y[shard_id * args.samples_per_shard:
                     (shard_id + 1) * args.samples_per_shard]

        np.save(os.path.join(
            args.shard_folder,
            f"gfp-x-{shard_id}.npy"), x_sliced)

        np.save(os.path.join(
            args.shard_folder,
            f"gfp-y-{shard_id}.npy"), y_sliced)
