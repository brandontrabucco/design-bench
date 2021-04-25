from design_bench import DATA_DIR
from design_bench import maybe_download
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import argparse
import os
import math


INVERSE_MAP = dict(a='t', t='a', c='g', g='c')


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw UTR")
    parser.add_argument("--shard-folder", type=str, default="./utr")
    parser.add_argument("--samples-per-shard", type=int, default=20000)
    parser.add_argument("--k-most-read", type=int, default=280000)
    args = parser.parse_args()

    # download the gfp dataset if not already
    maybe_download('1pRypiGVYl-kmJZaMhVbuA1PEvqauWBBM',
                   os.path.join(DATA_DIR, 'utr.zip'))
    utr_dir = os.path.join(DATA_DIR, 'utr')

    # load the static dataset and sort by total reads
    df = pd.read_csv(os.path.join(utr_dir, 'egfp_unmod_1.csv'))
    df.sort_values('total_reads', inplace=True, ascending=False)

    # select only the top k most read training examples
    df.reset_index(inplace=True, drop=True)
    df = df.iloc[:args.k_most_read]

    # load the 10 mer sequences from the dataset
    seq0 = np.array([list(x.lower()) for x in df["utr"].tolist()])
    seq1 = np.array([[INVERSE_MAP[c] for c in x] for x in seq0])
    x = np.concatenate([seq0, seq1], axis=0)

    # build an integer encoder for the allowed amino acids
    encoder = OrdinalEncoder(dtype=np.int32)
    encoder.fit(x.reshape((-1, 1)))

    # encode a dataset of peptide sequences into categorical features
    x = encoder.transform(x.reshape((-1, 1))).reshape(x.shape)

    # extract the ribosome loading score for each polypeptide
    ribosome_loading = df["rl"].to_numpy().reshape((-1, 1))

    # repeat the ribosome loading score twice because of DNA pair symmetry
    y = np.concatenate([ribosome_loading,
                        ribosome_loading], axis=0).astype(np.float32)

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
            f"utr-x-{shard_id}.npy"), x_sliced)

        np.save(os.path.join(
            args.shard_folder,
            f"utr-y-{shard_id}.npy"), y_sliced)
