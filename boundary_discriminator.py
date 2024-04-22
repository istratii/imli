import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, opening, square
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import Classes


class BoundaryDiscriminator:
    def __init__(self, rect, _class, hsi, lidar, groundtruth):
        self._hsi = hsi
        self._lidar = lidar
        self._class = _class
        self._rect = rect
        self._init_dataset(hsi, lidar, groundtruth)
        self._estimator = make_pipeline(
            StandardScaler(),
            LinearSVC(C=10, random_state=42, max_iter=2000, dual="auto"),
        )

    def _init_dataset(self, hsi, lidar, groundtruth, step=5):
        groundtruth_reshaped = groundtruth.reshape(-1)
        hsi_reshaped = hsi.reshape(-1, hsi.shape[-1])
        lidar_reshaped = lidar.reshape(-1, 1)
        mask_class = groundtruth_reshaped == self._class
        X = np.hstack((hsi_reshaped, lidar_reshaped))[mask_class]
        y = groundtruth_reshaped[mask_class]
        ay, ax, by, bx = self._rect
        hsi_bis = np.vstack(
            (
                hsi[:ay:step, ::step].reshape(-1, hsi.shape[-1]),
                hsi[by::step, ::step].reshape(-1, hsi.shape[-1]),
                hsi[ay:by:step, :ax:step].reshape(-1, hsi.shape[-1]),
                hsi[ay:by:step, bx::step].reshape(-1, hsi.shape[-1]),
            )
        )
        lidar_bis = np.hstack(
            (
                lidar[:ay:step, ::step].reshape(-1),
                lidar[by::step, ::step].reshape(-1),
                lidar[ay:by:step, :ax:step].reshape(-1),
                lidar[ay:by:step, bx::step].reshape(-1),
            )
        )
        X = np.vstack((X, np.hstack((hsi_bis, lidar_bis.reshape(-1, 1)))))
        y = np.hstack((y, np.zeros(lidar_bis.shape)))
        sss = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2)
        train_idx, test_idx = next(sss.split(X, y))
        self._X_train = X[train_idx]
        self._y_train = y[train_idx]
        self._X_test = X[test_idx]
        self._y_test = y[test_idx]

    def fit(self):
        return self._estimator.fit(self._X_train, self._y_train)

    def score(self):
        return self._estimator.score(self._X_test, self._y_test)

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_with_postprocessing(self, hsi, lidar, area_threshold=128):
        X = np.hstack((hsi.reshape(-1, hsi.shape[-1]), lidar.reshape(-1, 1)))
        y_pred = self.predict(X)
        img_pred = y_pred.reshape(hsi.shape[:-1])
        img_pred = closing(img_pred, square(3))
        img_pred = opening(img_pred)
        img_pred = label(img_pred)
        props = regionprops(img_pred)
        for prop in props:
            if prop.area < area_threshold:
                img_pred[img_pred == prop.label] = Classes.UNKNOWN
        img_pred[img_pred != Classes.UNKNOWN] = self._class
        return img_pred


if __name__ == "__main__":
    hsi = np.load("data/hyperspectral.npy")
    groundtruth = np.load("groundtruth/groundtruth.npy")
    lidar = np.load("data/lidar.npy")
    model = BoundaryDiscriminator(
        (100, 200, 250, 350), Classes.RUNNING_TRACK, hsi, lidar, groundtruth
    )
    model.fit()
    model.score()
