import numpy as np
from sklearn.svm import LinearSVC

from boundary_discriminator import BoundaryDiscriminator
from utils import Classes, load, save


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
    def model_name():
        return "boundary_discriminator_grass_synthetic"

    @staticmethod
    def fit(hsi, lidar, groundtruth):
        bd_grass_synthetic = BoundaryDiscriminator(
            (100, 200, 250, 350), Classes.GRASS_SYNTHETIC, hsi, lidar, groundtruth
        )
        bd_grass_synthetic.fit()
        assert bd_grass_synthetic.score() > 0.99
        save(bd_grass_synthetic, VegetationDiscriminator.model_name())

    @staticmethod
    def predict(hsi, lidar):
        ndvi = VegetationDiscriminator.ndvi(hsi)
        savi = VegetationDiscriminator.savi(hsi)
        model = load(VegetationDiscriminator.model_name())
        assert model is not None, "model was not found, call .fit()"
        features = np.hstack((hsi.reshape(-1, hsi.shape[-1]), lidar.reshape(-1, 1)))
        mask_synt = model.predict(features) == Classes.GRASS_SYNTHETIC
        mask_synt = mask_synt.reshape(hsi.shape[:-1])
        return ndvi | savi | mask_synt
