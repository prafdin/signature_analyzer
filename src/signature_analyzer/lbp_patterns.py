import math

import numpy as np
from skimage import feature
from PIL import ImageFilter
from PIL.Image import Image

from image_processor.image_processor import ImageProcessor, DefaultImageProcessorSettings
from signature_analyzer.utils import pad_2d_matrix_to_size, create_2d_square_rolling_window


def apply_LPB(image: np.ndarray):
    image = image / 255
    applied_window = create_2d_square_rolling_window(image, 3)
    matrices_list = applied_window.reshape(-1, 3, 3).tolist()
    lbp_matrix = np.array([
        [128, 1, 2],
        [64, 0, 4],
        [32, 16, 8]
    ])
    result = list(map(lambda m: np.sum(np.multiply(m, lbp_matrix)), matrices_list))
    return result


def calc_lbp_patterns_with_img_preprocess(calc_lbp_patterns_func):
    def preprocess_img_and_calc_lbp_patterns(image: Image):
        default_settings = DefaultImageProcessorSettings
        image = ImageProcessor.img_to_gray(image, default_settings)
        image = ImageProcessor.contrast_img(image, default_settings)
        # image = ImageProcessor.blur_img(image, default_settings)
        image = ImageProcessor.img_to_bin(image, default_settings)
        image = ImageProcessor.morph_open(image, default_settings)

        default_settings.dilate_iterations = 2
        image = ImageProcessor.dilate_img(image, default_settings)

        image = ImageProcessor.erode_img(image, default_settings)

        image = ImageProcessor.thin_img(image, default_settings)

        return calc_lbp_patterns_func(image)

    return preprocess_img_and_calc_lbp_patterns


@calc_lbp_patterns_with_img_preprocess
def calc_lbp_patterns(image: Image):
    LPB_patterns = feature.local_binary_pattern(image, 8, 1, method="uniform").ravel()
    LPB_patterns = list(map(lambda x: int(x), LPB_patterns))
    return LPB_patterns
