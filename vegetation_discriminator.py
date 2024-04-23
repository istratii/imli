import numpy as np
from sklearn.svm import LinearSVC

from boundary_discriminator import BoundaryDiscriminator
from utils import Classes, model_exists, model_load, model_save


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
        model_name = VegetationDiscriminator.model_name()
        if not model_exists(model_name):
            return
        bd_grass_synthetic = BoundaryDiscriminator(
            (100, 200, 250, 350), Classes.GRASS_SYNTHETIC, hsi, lidar, groundtruth
        )
        bd_grass_synthetic.fit()
        assert bd_grass_synthetic.score() > 0.99
        model_save(bd_grass_synthetic, model_name)

    @staticmethod
    def predict(hsi, lidar):
        ndvi = VegetationDiscriminator.ndvi(hsi)
        savi = VegetationDiscriminator.savi(hsi)
        model = model_load(VegetationDiscriminator.model_name())
        assert model is not None, "model was not found, call .fit()"
        mask_synt = model.predict_with_postprocessing(hsi, lidar)
        mask_synt = mask_synt == Classes.GRASS_SYNTHETIC
        mask_synt = mask_synt.reshape(hsi.shape[:-1])
        return ndvi | savi | mask_synt
