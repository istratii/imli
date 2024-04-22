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
    def predict(hsi):
        ndvi = VegetationDiscriminator.ndvi(hsi)
        savi = VegetationDiscriminator.savi(hsi)
        return ndvi | savi
