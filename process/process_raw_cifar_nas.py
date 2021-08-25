import numpy as np
import argparse
import os
import itertools
import pandas as pd
import glob


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw NASBench")
    parser.add_argument("--csv-pattern",
                        type=str, default="*.csv")
    parser.add_argument("--shard-folder",
                        type=str, default="./")
    parser.add_argument("--samples-per-shard",
                        type=int, default=50000)
    args = parser.parse_args()
    files_list = []
    os.makedirs(os.path.join(
        args.shard_folder, f"cifar_nas/"), exist_ok=True)

    csv_files = glob.glob(args.csv_pattern)
    data = pd.concat(pd.read_csv(f) for f in csv_files)

    # loop through all possible architectures
    x_list = []
    y_list = []
    shard_id = 0
    for index, row in data.iterrows():

        x = np.array([int(value) for value in row['arch_config'].split('-')])
        y = np.array([float(row['test_accuracy'])])

        # save the x and y values and potentially write a shard
        x_list.append(x)
        y_list.append(y)
        if len(y_list) == args.samples_per_shard:

            x_shard = np.stack(x_list, axis=0).astype(np.int32)
            x_shard_path = os.path.join(
                args.shard_folder, f"cifar_nas/cifar_nas-x-{shard_id}.npy")
            np.save(x_shard_path, x_shard)

            y_shard = np.stack(y_list, axis=0).astype(np.float32)
            y_shard_path = os.path.join(
                args.shard_folder, f"cifar_nas/cifar_nas-y-{shard_id}.npy")
            np.save(y_shard_path, y_shard)
            files_list.append(f"cifar_nas/cifar_nas-x-{shard_id}.npy")

            x_list = []
            y_list = []
            shard_id += 1

    # serialize another shard if there are any remaining
    if len(x_list) > 0:

        x_shard = np.stack(x_list, axis=0).astype(np.int32)
        x_shard_path = os.path.join(
            args.shard_folder, f"cifar_nas/cifar_nas-x-{shard_id}.npy")
        np.save(x_shard_path, x_shard)

        y_shard = np.stack(y_list, axis=0).astype(np.float32)
        y_shard_path = os.path.join(
            args.shard_folder, f"cifar_nas/cifar_nas-y-{shard_id}.npy")
        np.save(y_shard_path, y_shard)
        files_list.append(f"cifar_nas/cifar_nas-x-{shard_id}.npy")

    print(files_list)
