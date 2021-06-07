import numpy as np
import argparse
import os
import itertools


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Toy Discrete Dataset")
    parser.add_argument("--shard-folder", type=str, default="./")
    parser.add_argument("--seq-length", type=int, default=8)
    parser.add_argument("--options", type=int, default=4)
    parser.add_argument("--samples-per-shard", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(os.path.join(
        args.shard_folder, f"toy_discrete/"), exist_ok=True)

    xs = []
    ys = []
    files_list = []
    shard_id = 0

    options = list(range(args.options))
    list_options = [options for i in range(args.seq_length)]
    for sample in itertools.product(*list_options):

        x = np.array(sample, dtype=np.int32)
        y = -np.square(x.astype(
            np.float32)).sum(keepdims=True).astype(np.float32)

        xs.append(x)
        ys.append(y)

        if len(xs) == args.samples_per_shard:

            np.save(os.path.join(
                args.shard_folder,
                f"toy_discrete/"
                f"toy_discrete-x-{shard_id}.npy"), xs)

            np.save(os.path.join(
                args.shard_folder,
                f"toy_discrete/"
                f"toy_discrete-y-{shard_id}.npy"), ys)

            xs = []
            ys = []
            files_list.append(f"toy_discrete/"
                              f"toy_discrete-x-{shard_id}.npy")
            shard_id += 1

    if len(xs) > 0:

        np.save(os.path.join(
            args.shard_folder,
            f"toy_discrete/"
            f"toy_discrete-x-{shard_id}.npy"), xs)

        np.save(os.path.join(
            args.shard_folder,
            f"toy_discrete/"
            f"toy_discrete-y-{shard_id}.npy"), ys)

        xs = []
        ys = []
        files_list.append(f"toy_discrete/"
                          f"toy_discrete-x-{shard_id}.npy")
        shard_id += 1
