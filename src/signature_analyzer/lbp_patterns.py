import math

import numpy as np
from PIL import ImageFilter
from PIL.Image import Image

from signature_analyzer.utils import img_to_gray, img_to_bin, pad_2d_matrix_to_size, create_2d_square_rolling_window
from signature_analyzer.zhang_suen import thinning

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

def calc_lbp_patterns(image: Image):
    window_size = 3
    gray_img = img_to_gray(image)
    blured_gray_img = gray_img.filter(filter=ImageFilter.BLUR)
    bin_img = img_to_bin(blured_gray_img, threshold=240)
    thinned_img = thinning(bin_img)
    new_x = window_size * (math.ceil(thinned_img.shape[0] / window_size))
    new_y = window_size * (math.ceil(thinned_img.shape[1] / window_size))
    thinned_img = pad_2d_matrix_to_size(thinned_img, new_x, new_y)
    LPB_patterns = apply_LPB(thinned_img)
    LPB_patterns = list(map(lambda x: int(x), LPB_patterns))
    # LPB_patterns = list(filter(lambda x: x != 0, LPB_patterns)) # Exclude zero patterns
    return LPB_patterns
