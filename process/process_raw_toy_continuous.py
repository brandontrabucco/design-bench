from design_bench.disk_resource import DATA_DIR
from design_bench.disk_resource import google_drive_download
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
import pandas as pd
import numpy as np
import argparse
import glob
import os
import math
import itertools


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Toy Continuous Dataset")
    parser.add_argument("--shard-folder", type=str, default="./")
    parser.add_argument("--seq-length", type=int, default=8)
    parser.add_argument("--options", type=int, default=4)
    parser.add_argument("--samples-per-shard", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(os.path.join(
        args.shard_folder, f"toy_continuous/"), exist_ok=True)

    xs = []
    ys = []
    files_list = []
    shard_id = 0

    options = list(range(args.options))
    list_options = [options for i in range(args.seq_length)]
    for sample in itertools.product(*list_options):

        x = np.array(sample, dtype=np.int32).astype(np.float32)
        x = (x + np.random.uniform(0., 1., size=x.shape)) / args.options
        y = -np.square(x - 0.5).sum(keepdims=True).astype(np.float32)

        xs.append(x)
        ys.append(y)

        if len(xs) == args.samples_per_shard:

            np.save(os.path.join(
                args.shard_folder,
                f"toy_continuous/"
                f"toy_continuous-x-{shard_id}.npy"), xs)

            np.save(os.path.join(
                args.shard_folder,
                f"toy_continuous/"
                f"toy_continuous-y-{shard_id}.npy"), ys)

            xs = []
            ys = []
            files_list.append(f"toy_continuous/"
                              f"toy_continuous-x-{shard_id}.npy")
            shard_id += 1

    if len(xs) > 0:

        np.save(os.path.join(
            args.shard_folder,
            f"toy_continuous/"
            f"toy_continuous-x-{shard_id}.npy"), xs)

        np.save(os.path.join(
            args.shard_folder,
            f"toy_continuous/"
            f"toy_continuous-y-{shard_id}.npy"), ys)

        xs = []
        ys = []
        files_list.append(f"toy_continuous/"
                          f"toy_continuous-x-{shard_id}.npy")
        shard_id += 1
