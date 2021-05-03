from design_bench.disk_resource import DATA_DIR
from design_bench.disk_resource import google_drive_download
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import argparse
import os
import math
import glob
import zipfile


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw TF Binding 8")
    parser.add_argument("--shard-folder", type=str, default="./")
    parser.add_argument("--samples-per-shard", type=int, default=100000)
    args = parser.parse_args()

    # download the tf_bind_8 dataset if not already present
    target = os.path.join(DATA_DIR, 'TF_binding_landscapes.zip')
    google_drive_download('1xS6N5qSwyFLC-ZPTADYrxZuPHjBkZCrj',
                          target)
    with zipfile.ZipFile(target, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(target))

    # load the static dataset
    tf_dir = os.path.join(os.path.join(
        DATA_DIR, 'TF_binding_landscapes'), 'landscapes')

    all_files = glob.glob(os.path.join(tf_dir, "*_8mers.txt"))
    transcription_factors = [os.path.basename(
        f).replace("_8mers.txt", "") for f in all_files]

    os.makedirs(args.shard_folder, exist_ok=True)
    files_list = []
    for transcription_factor in transcription_factors:

        data = pd.read_csv(os.path.join(
            tf_dir, f'{transcription_factor}_8mers.txt'), sep="\t")

        # load the 8 mer sequences from the dataset
        seq0 = np.array([list(x.lower()) for x in data["8-mer"].tolist()])
        seq1 = np.array([list(x.lower()) for x in data["8-mer.1"].tolist()])
        x = np.concatenate([seq0, seq1], axis=0)

        # build an integer encoder for the allowed amino acids
        encoder = OrdinalEncoder(dtype=np.int32)
        encoder.fit(x.reshape((-1, 1)))

        # encode a dataset of peptide sequences into categorical features
        x = encoder.transform(x.reshape((-1, 1))).reshape(x.shape)

        # extract the enrichment score for each polypeptide
        enrichment = data["E-score"].to_numpy().reshape((-1, 1))
        enrichment = (enrichment - enrichment.min()) / \
                     (enrichment.max() - enrichment.min())

        # repeat the enrichment score twice because of DNA pair symmetry
        y = np.concatenate([
            enrichment, enrichment], axis=0).astype(np.float32)

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
                f"tf_bind_8-{transcription_factor}/"), exist_ok=True)
            files_list.append(f"tf_bind_8-{transcription_factor}/"
                              f"tf_bind_8-x-{shard_id}.npy")
            np.save(os.path.join(
                args.shard_folder,
                f"tf_bind_8-{transcription_factor}/"
                f"tf_bind_8-x-{shard_id}.npy"), x_sliced)
            np.save(os.path.join(
                args.shard_folder,
                f"tf_bind_8-{transcription_factor}/"
                f"tf_bind_8-y-{shard_id}.npy"), y_sliced)

    print(files_list)
