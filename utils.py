import gzip
import os
import pickle
from enum import Enum

import numpy as np


class Classes(int, Enum):
    UNKNOWN = 0
    GRASS_HEALTHY = 1
    GRASS_STRESSED = 2
    GRASS_SYNTHETIC = 3
    TREE = 4
    SOIL = 5
    WATER = 6
    RESIDENTIAL = 7
    COMMERCIAL = 8
    ROAD = 9
    HIGHWAY = 10
    RAILWAY = 11
    PARKING_LOT1 = 12
    PARKING_LOT2 = 13
    TENNIS_COURT = 14
    RUNNING_TRACK = 15


def cmap_get():
    from matplotlib.colors import ListedColormap

    return ListedColormap(
        np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.80392157, 0.0]],
                [[0.49803922, 1.0, 0.0]],
                [[0.18039216, 0.80392157, 0.34117647]],
                [[0.0, 0.54509804, 0.0]],
                [[0.62745098, 0.32156863, 0.17647059]],
                [[0.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0]],
                [[0.84705882, 0.74901961, 0.84705882]],
                [[1.0, 0.0, 0.0]],
                [[0.54509804, 0.0, 0.0]],
                [[0.39215686, 0.39215686, 0.39215686]],
                [[1.0, 1.0, 0.0]],
                [[0.93333333, 0.60392157, 0.0]],
                [[0.33333333, 0.10196078, 0.54509804]],
                [[1.0, 0.49803922, 0.31372549]],
            ]
        )
    )


def data_get():
    """Returns X, y"""

    groundtruth = np.load("groundtruth/groundtruth.npy").reshape(-1)
    hsi = np.load("data/hyperspectral.npy")
    hsi = hsi.reshape(-1, hsi.shape[-1])
    lidar = np.load("data/lidar.npy").reshape(-1)
    idx = groundtruth.nonzero()[0]

    X = np.hstack((hsi[idx], lidar[idx].reshape((-1, 1))))
    y = groundtruth[idx]

    return X, y


def model_save(obj, name):
    with gzip.open(f"data/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def model_load(name):
    try:
        with gzip.open(f"data/{name}.pkl", "rb") as f:
            return pickle.load(f)
    except:
        pass


def model_exists(name):
    return os.path.isfile(f"data/{name}.pkl")


def compute_coincidence_with_groundtruth(groundtruth, predicted, _class):
    m = groundtruth == _class
    return np.sum(predicted[m] == _class) / np.sum(m)
