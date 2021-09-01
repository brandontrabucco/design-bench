import pickle as pkl
import design_bench as db
import numpy as np
from design_bench.oracles.feature_extractors.morgan_fingerprint_features import MorganFingerprintFeatures


if __name__ == "__main__":

    with open("type_assay_pairs.pkl", "rb") as f:
        type_assay_pairs = pkl.load(f)

    all_rank_corr = []

    for type_name, assay_name in type_assay_pairs:

        task = db.make(
            'ChEMBLMorganFingerprint-FullyConnected-v0',

            dataset_kwargs=dict(
                max_samples=None,
                distribution=None,
                max_percentile=50,
                min_percentile=0,
                assay_chembl_id=assay_name,
                standard_type=type_name),

            oracle_kwargs=dict(
                noise_std=0.0,
                max_samples=None,
                distribution=None,
                max_percentile=100,
                min_percentile=0,

                feature_extractor=MorganFingerprintFeatures(dtype=np.float32),

                model_kwargs=dict(
                    hidden_size=512,
                    activation='relu',
                    num_layers=2,
                    epochs=5,
                    shuffle_buffer=5000,
                    learning_rate=0.0001),

                split_kwargs=dict(
                    val_fraction=0.1,
                    subset=None,
                    shard_size=50000,
                    to_disk=True,
                    disk_target=f"chembl-{type_name}-{assay_name}/split",
                    is_absolute=False))

        )

        print(type_name, assay_name,
              task.oracle.params['rank_correlation'])

        all_rank_corr.append(task.oracle.params['rank_correlation'])

    best_type_name, best_assay_name = \
        type_assay_pairs[np.argmax(np.array(all_rank_corr))]
