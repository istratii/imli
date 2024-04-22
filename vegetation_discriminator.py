import numpy as np
from sklearn.svm import LinearSVC

from boundary_discriminator import BoundaryDiscriminator
from classifier import Classifier
from utils import Classes, data_get


class VegetationDiscriminator:
    @staticmethod
    def ndvi(hsi, threshold=0):
        nir = (hsi[:, :, 104] + hsi[:, :, 105]) / 2
        red = (hsi[:, :, 60] + hsi[:, :, 61]) / 2
        ndvi = (nir - red) / (nir + red + 1e-9)
        return ndvi > threshold

    @staticmethod
    def savi(hsi, threshold=0):
        nir = (hsi[:, :, 104] + hsi[:, :, 105]) / 2
        red = (hsi[:, :, 60] + hsi[:, :, 61]) / 2
        savi = (1.5 * (nir - red)) / (nir + red + 0.5 + 1e-9)
        return savi > threshold

    @staticmethod
    def discriminate_grass_synthetic(hsi, lidar, groundtruth):
        bd_grass_synthetic = BoundaryDiscriminator(
            (100, 200, 250, 350), Classes.GRASS_SYNTHETIC, hsi, lidar, groundtruth
        )
        bd_grass_synthetic.fit()
        assert bd_grass_synthetic.score() > 0.99
        features = np.hstack((hsi.reshape(-1, hsi.shape[-1]), lidar.reshape(-1, 1)))
        mask = bd_grass_synthetic.predict(features) == Classes.GRASS_SYNTHETIC
        mask = mask.reshape(groundtruth.shape)
        return mask

    @staticmethod
    def predict(hsi, lidar, groundtruth):
        ndvi = VegetationDiscriminator.ndvi(hsi)
        savi = VegetationDiscriminator.savi(hsi)
        mask_synthetic = VegetationDiscriminator.discriminate_grass_synthetic(
            hsi, lidar, groundtruth
        )
        return ndvi | savi | mask_synthetic
