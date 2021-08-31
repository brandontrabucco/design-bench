from design_bench.disk_resource import DATA_DIR
from design_bench.disk_resource import google_drive_download
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
import pandas as pd
import numpy as np
import argparse
import glob
import os
import math


COLUMNS = [
    'Molecule ChEMBL ID', 'Molecule Name', 'Molecule Max Phase',
    'Molecular Weight', '#RO5 Violations', 'AlogP', 'Compound Key',
    'Smiles', 'Standard Type', 'Standard Relation', 'Standard Value',
    'Standard Units', 'pChEMBL Value', 'Data Validity Comment', 'Comment',
    'Uo Units', 'Ligand Efficiency BEI', 'Ligand Efficiency LE',
    'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 'Potential Duplicate',
    'Assay ChEMBL ID', 'Assay Description', 'Assay Type', 'BAO Format ID',
    'BAO Label', 'Assay Organism', 'Assay Tissue ChEMBL ID',
    'Assay Tissue Name', 'Assay Cell Type', 'Assay Subcellular Fraction',
    'Assay Parameters', 'Assay Variant Accession', 'Assay Variant Mutation',
    'Target ChEMBL ID', 'Target Name', 'Target Organism', 'Target Type',
    'Document ChEMBL ID', 'Source ID', 'Source Description',
    'Document Journal', 'Document Year', 'Cell ChEMBL ID', 'Properties']


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw ChEMBL")
    parser.add_argument("--dir", type=str, default="data/chembl_activities")
    parser.add_argument("--shard-folder", type=str, default="./")
    parser.add_argument("--max-smiles-len", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--samples-per-shard", type=int, default=50000)
    args = parser.parse_args()

    file_matches = glob.glob(os.path.join(args.dir, "*.csv"))
    data = None
    for match in sorted(file_matches):
        print(f"reading csv file: {match}")

        df = pd.read_csv(match, sep=";")
        df.columns = COLUMNS

        df = df.dropna(subset=["Standard Value", "Smiles"])
        df = df[["Molecule ChEMBL ID", "Molecule Name", "Smiles",
                 "Standard Type", "Standard Value", "Standard Units",
                 "Assay ChEMBL ID", "Assay Description", "Assay Type"]]

        data = df if data is None else data.append(df, ignore_index=True)

    data = data.dropna(subset=['Standard Value'])
    data = data.groupby(["Standard Type", "Assay ChEMBL ID"])\
        .filter(lambda x: sum([1 if len(sm) < args.max_smiles_len
                               else 0 for sm in x['Smiles'].tolist()]) >= args.min_samples)

    group_sizes = data.groupby(["Standard Type", "Assay ChEMBL ID"]).size()
    group = list(zip(*list(
        data.groupby(["Standard Type", "Assay ChEMBL ID"]))))[0]

    # download the molecule dataset if not already
    google_drive_download('1u5wQVwVSK7PG6dxGL2p_6pXf8gvsfUAk',
                          os.path.join(DATA_DIR, 'smiles_vocab.txt'))
    tokenizer = SmilesTokenizer(
        os.path.join(DATA_DIR, 'smiles_vocab.txt'))

    os.makedirs(args.shard_folder, exist_ok=True)
    files_list = []
    for standard_type, assay_chembl_id in group:

        # load the static dataset
        df = data

        # remove all measurements for different standard type and assays
        df = df[df["Standard Type"] ==
                standard_type][df["Assay ChEMBL ID"] == assay_chembl_id]

        x = df['Smiles'].to_list()
        y = df['Standard Value'].to_list()
        x, y = zip(*[(xi, yi) for xi, yi in
                     zip(x, y) if len(xi) < args.max_smiles_len])

        # build an integer encoder for smiles sequences
        x = tokenizer(x, padding="longest")["input_ids"]
        x = np.array(x).astype(np.int32)

        # extract the prediction property of interest
        y = np.array(y).astype(np.float32).reshape([-1, 1])

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
                f"chembl-{standard_type}-{assay_chembl_id}"), exist_ok=True)
            files_list.append(f"chembl-{standard_type}-{assay_chembl_id}/"
                              f"chembl-x-{shard_id}.npy")
            np.save(os.path.join(
                args.shard_folder,
                f"chembl-{standard_type}-{assay_chembl_id}/"
                f"chembl-x-{shard_id}.npy"), x_sliced)
            np.save(os.path.join(
                args.shard_folder,
                f"chembl-{standard_type}-{assay_chembl_id}/"
                f"chembl-y-{shard_id}.npy"), y_sliced)

            print(f"chembl-{standard_type}-{assay_chembl_id}/"
                  f"chembl-x-{shard_id}.npy", x_sliced.shape[0])

    print(files_list)
