from design_bench.disk_resource import DATA_DIR
from design_bench.disk_resource import google_drive_download
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import argparse
import os
import math


INVERSE_MAP = dict(a='t', t='a', c='g', g='c')


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw TF Binding 10")
    parser.add_argument("--shard-folder", type=str, default="./")
    parser.add_argument("--samples-per-shard", type=int, default=100000)
    args = parser.parse_args()

    # download the tf_bind_10 dataset if not already present
    google_drive_download('1qeX6vnuOLdj8tzyQ3ub1AjSk4kVlUUQb',
                          os.path.join(DATA_DIR, 'tfbind10_counts.txt'))

    # load experimentally determined binding stability
    data_file = os.path.join(DATA_DIR, 'tfbind10_counts.txt')
    df = pd.read_csv(data_file, sep="\t")
    transcription_factors = df['protein'].unique().tolist()

    os.makedirs(args.shard_folder, exist_ok=True)
    files_list = []
    for transcription_factor in transcription_factors:

        data = df.loc[df['protein'] == transcription_factor]

        # filter to replace infinite ddG values with bounds
        filtered = data.loc[data['ddG'] != np.inf]
        filtered = filtered.loc[data['ddG'] != -np.inf, 'ddG']
        data['ddG'].replace(np.inf, filtered.max(), inplace=True)
        data['ddG'].replace(-np.inf, filtered.min(), inplace=True)

        # load the 10 mer sequences from the dataset
        seq0 = np.array([list(x.lower()) for x in data["flank"].tolist()])
        seq1 = np.array([[INVERSE_MAP[c] for c in x] for x in seq0])
        x = np.concatenate([seq0, seq1], axis=0)

        # build an integer encoder for the allowed amino acids
        encoder = OrdinalEncoder(dtype=np.int32)
        encoder.fit(x.reshape((-1, 1)))

        # encode a dataset of peptide sequences into categorical features
        x = encoder.transform(x.reshape((-1, 1))).reshape(x.shape)

        # extract the delta delta g score for each polypeptide
        ddG = data["ddG"].to_numpy().reshape((-1, 1))

        # repeat the delta delta g score twice because of DNA pair symmetry
        y = np.concatenate([
            ddG, ddG], axis=0).astype(np.float32)

        # calculate the number of batches per single shard
        batch_per_shard = int(math.ceil(
            y.shape[0] / args.samples_per_shard))

        # loop once per batch contained in the shard
        for shard_id in range(batch_per_shard):

            # slice out a component of the current shard
            x_sliced = x[shard_id * args.samples_per_shard:
                         (shard_id + 1) * args.samples_per_shard]
            y_sliced = y[shard_id * args.samples_per_shard:
                         (shard_id + 1) * args.samples_per_shard]

            os.makedirs(os.path.join(
                args.shard_folder,
                f"tf_bind_10-{transcription_factor}/"), exist_ok=True)
            files_list.append(f"tf_bind_10-{transcription_factor}/"
                              f"tf_bind_10-x-{shard_id}.npy")
            np.save(os.path.join(
                args.shard_folder,
                f"tf_bind_10-{transcription_factor}/"
                f"tf_bind_10-x-{shard_id}.npy"), x_sliced)
            np.save(os.path.join(
                args.shard_folder,
                f"tf_bind_10-{transcription_factor}"
                f"/tf_bind_10-y-{shard_id}.npy"), y_sliced)

    print(files_list)
