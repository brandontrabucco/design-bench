from design_bench.disk_resource import DATA_DIR
from design_bench.disk_resource import google_drive_download
import pandas as pd
import numpy as np
import argparse
import os
import math


# order is important due to BLOSUM matrix in ProteinKernel
AA = ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h',
      'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v']
AA_IDX = {AA[i]: i for i in range(len(AA))}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw GFP")
    parser.add_argument("--shard-folder", type=str, default="./")
    parser.add_argument("--samples-per-shard", type=int, default=5000)
    args = parser.parse_args()

    # download the gfp dataset if not already
    google_drive_download('1_jcPkQ-M1FRhkEONoE57WEbp_Rivkho2',
                          os.path.join(DATA_DIR, 'gfp_data.csv'))

    # load the static dataset
    df = pd.read_csv(os.path.join(DATA_DIR, 'gfp_data.csv'))

    # remove all proteins with a stop marker
    df = df.loc[df.loc[
        ~df['aaSequence'].str.contains('!')].index]

    # extract the amino acid sequences for each protein
    x = np.array([list(x) for x in
                  df['aaSequence'].to_list()])

    # encode a dataset of amino acid sequences into categorical features
    x = np.array([[AA_IDX[token.lower()]
                   for token in row] for row in x], dtype=np.int32)

    # format the fluorescence values to a tensor
    y = df['medianBrightness'] \
        .to_numpy().astype(np.float32).reshape([-1, 1])

    # calculate the number of batches per single shard
    batch_per_shard = int(math.ceil(
        y.shape[0] / args.samples_per_shard))

    # loop once per batch contained in the shard
    os.makedirs(os.path.join(
        args.shard_folder, f"gfp/"), exist_ok=True)
    files_list = []
    for shard_id in range(batch_per_shard):

        # slice out a component of the current shard
        x_sliced = x[shard_id * args.samples_per_shard:
                     (shard_id + 1) * args.samples_per_shard]
        y_sliced = y[shard_id * args.samples_per_shard:
                     (shard_id + 1) * args.samples_per_shard]

        files_list.append(f"gfp/gfp-x-{shard_id}.npy")
        np.save(os.path.join(
            args.shard_folder,
            f"gfp/gfp-x-{shard_id}.npy"), x_sliced)
        np.save(os.path.join(
            args.shard_folder,
            f"gfp/gfp-y-{shard_id}.npy"), y_sliced)

    print(files_list)
