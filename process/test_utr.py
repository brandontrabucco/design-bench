from design_bench.disk_resource import DATA_DIR
from design_bench.disk_resource import google_drive_download
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import scipy.stats as stats
import pandas as pd
import numpy as np
import argparse
import os
import math

import tensorflow.keras as keras
np.random.seed(1337)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D


INVERSE_MAP = dict(a='t', t='a', c='g', g='c')


def train_model(x, y, border_mode='same', inp_len=50, nodes=40,
                layers=3, filter_len=8, nbr_filters=120,
                dropout1=0., dropout2=0., dropout3=0., nb_epoch=3):

    ''' Build model archicture and fit.'''

    model = Sequential()
    model.add(Embedding(4, nbr_filters, input_shape=(inp_len,)))
    if layers >= 1:
        model.add(Conv1D(activation="relu",
                         padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
    if layers >= 2:
        model.add(Conv1D(activation="relu",
                         padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
        model.add(Dropout(dropout1))
    if layers >= 3:
        model.add(Conv1D(activation="relu",
                         padding=border_mode, filters=nbr_filters,
                         kernel_size=filter_len))
        model.add(Dropout(dropout2))
    model.add(Flatten())

    model.add(Dense(1))
    model.add(Activation('linear'))

    # compile the model
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9,
                                 beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=adam)

    model.fit(x, y, batch_size=128, epochs=nb_epoch, verbose=1)
    return model


def test_data(df, model, test_seq, obs_col, output_col='pred'):
    '''Predict mean ribosome load using model and test set UTRs'''

    # Scale the test set mean ribosome load
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].to_numpy().reshape(-1, 1))

    # Make predictions
    predictions = model.predict(test_seq).reshape(-1)

    # Inverse scaled predicted mean ribosome load and return in a column labeled 'pred'
    df.loc[:, output_col] = scaler.inverse_transform(predictions)
    return df


def one_hot_encode(df, col='utr', seq_len=50):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0],
             'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors


def r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


nuc_d = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0],
         2: [0, 0, 1, 0], 3: [0, 0, 0, 1], 4: [0, 0, 0, 0]}


if __name__ == "__main__":

    from design_bench.datasets.discrete.utr_dataset import UTRDataset

    dataset = UTRDataset()

    dataset.map_normalize_y()

    x = dataset.x
    y = dataset.y

    model = train_model(x, y, nb_epoch=3, border_mode='same', inp_len=50,
                        nodes=40, layers=3, nbr_filters=120, filter_len=8,
                        dropout1=0, dropout2=0, dropout3=0.2)

