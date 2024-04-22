import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import binary_dilation, square

from boundary_discriminator import BoundaryDiscriminator
from utils import Classes, load, save


class WaterDescriminator:
    @staticmethod
    def ndwi(hsi, threshold):
        green = (hsi[:, :, 38] + hsi[:, :, 39]) / 2
        nir = (hsi[:, :, 104] + hsi[:, :, 105]) / 2
        ndwi = (green - nir) / (green + nir + 1e-9)
        return ndwi > threshold

    @staticmethod
    def model_name():
        return "boundary_discriminator_water"

    @staticmethod
    def fit(hsi, lidar, groundtruth):
        bd_water = BoundaryDiscriminator(
            (10, 700, 390, 1200), Classes.WATER, hsi, lidar, groundtruth
        )
        bd_water.fit()
        assert bd_water.score() > 0.99
        save(bd_water, WaterDescriminator.model_name())

    @staticmethod
    def predict(hsi, lidar):
        model = load(WaterDescriminator.model_name())
        assert model is not None, f"model not found, call .fit()"
        features = np.hstack((hsi.reshape(-1, hsi.shape[-1]), lidar.reshape(-1, 1)))
        mask = model.predict(features) == Classes.WATER
        mask = mask.reshape(hsi.shape[:-1])
        mask = binary_dilation(mask, square(5))
        ndwi = WaterDescriminator.ndwi(hsi, 0.3)
        mask = mask & ndwi
        mask = median_filter(mask, 3)
        return mask
