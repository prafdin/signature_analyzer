import math

import numpy as np
from PIL import ImageFilter
from PIL.Image import Image

from image_processor.image_processor import ImageProcessor, DefaultImageProcessorSettings
from signature_analyzer.utils import pad_2d_matrix_to_size, create_2d_square_rolling_window


def apply_LPB(image: np.ndarray):
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
        default_settings.bin_threshold = 240

        image = ImageProcessor.img_to_gray(image, default_settings)
        # image = ImageProcessor.blur_img(image, default_settings)
        image = ImageProcessor.img_to_bin(image, default_settings)
        image = ImageProcessor.thin_img(image, default_settings)

        return calc_lbp_patterns_func(image)

    return preprocess_img_and_calc_lbp_patterns


@calc_lbp_patterns_with_img_preprocess
def calc_lbp_patterns(image: Image):
    window_size = 3
    new_x = window_size * (math.ceil(image.shape[0] / window_size))
    new_y = window_size * (math.ceil(image.shape[1] / window_size))
    image = pad_2d_matrix_to_size(image, new_x, new_y)
    LPB_patterns = apply_LPB(image)
    LPB_patterns = list(map(lambda x: int(x), LPB_patterns))
    # LPB_patterns = list(filter(lambda x: x != 0, LPB_patterns)) # Exclude zero patterns
    return LPB_patterns
