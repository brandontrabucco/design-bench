from design_bench.disk_resource import DATA_DIR
from design_bench.disk_resource import google_drive_download
import pandas as pd
import numpy as np
import argparse
import os
import math


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw Superconductor")
    parser.add_argument("--shard-folder", type=str, default="./")
    parser.add_argument("--samples-per-shard", type=int, default=5000)
    args = parser.parse_args()

    # download the gfp dataset if not already
    google_drive_download('1AguXqbNrSc665sablzVJh4RHLodeXglx',
                          os.path.join(DATA_DIR, 'superconductor_unique_m.csv'))

    # load the static dataset
    df = pd.read_csv(os.path.join(
        DATA_DIR, 'superconductor_unique_m.csv'))

    # extract the relative mix of chemicals whose composition leads
    # to a superconducting material as a particular critical temperature
    x = df[df.columns[:-2]].to_numpy(dtype=np.float32)

    # extract the critical temperatures for each material
    y = df["critical_temp"] \
        .to_numpy(dtype=np.float32).reshape((-1, 1))

    # calculate the number of batches per single shard
    batch_per_shard = int(math.ceil(
        y.shape[0] / args.samples_per_shard))

    # loop once per batch contained in the shard

    os.makedirs(os.path.join(
            args.shard_folder,
            f"superconductor/"), exist_ok=True)
    files_list = []
    for shard_id in range(batch_per_shard):

        # slice out a component of the current shard
        x_sliced = x[shard_id * args.samples_per_shard:
                     (shard_id + 1) * args.samples_per_shard]
        y_sliced = y[shard_id * args.samples_per_shard:
                     (shard_id + 1) * args.samples_per_shard]

        files_list.append(f"superconductor/superconductor-x-{shard_id}.npy")
        np.save(os.path.join(
            args.shard_folder,
            f"superconductor/superconductor-x-{shard_id}.npy"), x_sliced)
        np.save(os.path.join(
            args.shard_folder,
            f"superconductor/superconductor-y-{shard_id}.npy"), y_sliced)

    print(files_list)
