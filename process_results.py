import pickle as pkl
import numpy as np
from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset


if __name__ == "__main__":

    with open('type_assay_pairs.pkl', 'rb') as f:
        type_assay_pairs = pkl.load(f)

    with open('results.txt', 'r') as f:
        all_results = f.readlines()

    type_assay_pairs_to_y_lr = dict()
    type_assay_pairs_to_y_rf = dict()

    type_assay_pairs_to_log_y_lr = dict()
    type_assay_pairs_to_log_y_rf = dict()

    descriptors = []
    values = []

    for row in all_results:
        row = row.split(',')
        if len(row) == 4:
            idx, log_y, lr_corr, rf_corr = row

            idx = int(idx.replace("Index: ", "").strip())
            log_y = log_y.replace("log_y: ", "").strip() == "True"
            lr_corr = float(lr_corr.replace("lr_corr: ", "").strip())
            rf_corr = float(rf_corr.replace("rf_corr: ", "").strip())

            standard_type, assay = type_assay_pairs[idx]
            dataset = ChEMBLDataset(assay_chembl_id=assay, standard_type=standard_type)
            size = dataset.dataset_size

            print(standard_type, assay, size, log_y, lr_corr, rf_corr)
            descriptors.append((standard_type, assay, size, log_y, 'lr_corr'))
            descriptors.append((standard_type, assay, size, log_y, 'rf_corr'))
            values.append(lr_corr)
            values.append(rf_corr)

            if log_y:
                type_assay_pairs_to_log_y_lr[(standard_type, assay)] = lr_corr
                type_assay_pairs_to_log_y_rf[(standard_type, assay)] = rf_corr

            else:
                type_assay_pairs_to_y_lr[(standard_type, assay)] = lr_corr
                type_assay_pairs_to_y_rf[(standard_type, assay)] = rf_corr

    with open('type_assay_pairs_to_y_lr.pkl', 'wb') as f:
        pkl.dump(type_assay_pairs_to_y_lr, f)

    with open('type_assay_pairs_to_y_rf.pkl', 'wb') as f:
        pkl.dump(type_assay_pairs_to_y_rf, f)

    with open('type_assay_pairs_to_log_y_lr.pkl', 'wb') as f:
        pkl.dump(type_assay_pairs_to_log_y_lr, f)

    with open('type_assay_pairs_to_log_y_rf.pkl', 'wb') as f:
        pkl.dump(type_assay_pairs_to_log_y_rf, f)

    top_idx = np.argsort(values)[::-1]
    for idx in top_idx:
        print(descriptors[idx], values[idx])

