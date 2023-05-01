import PIL.Image
import numpy as np
import cv2 as cv





def pad_2d_matrix_to_size(a: np.ndarray, new_xsize, new_ysize, pad_value=0):
    if (len(a.shape) != 2):
        raise Exception("Function works only with 2d matrix")

    current_xsize = a.shape[0]
    current_ysize = a.shape[1]
    delta_x = new_xsize - current_xsize
    delta_y = new_ysize - current_ysize

    if (delta_x < 0 or delta_y < 0):
        raise Exception("The current size of the matrix is greater than the new size")
    d = int(delta_y // 2),
    d = int(delta_y - (delta_y // 2))
    return np.pad(
        a,
        (
            (
                int(delta_x // 2),
                int(delta_x - (delta_x // 2))
            ),
            (
                int(delta_y // 2),
                int(delta_y - (delta_y // 2))
            ),
        ),
        mode='constant',
        constant_values=pad_value
    )


# https://habr.com/ru/articles/489734/#2d
def rolling_window_2d(a, window_shape, dx=1, dy=1):
    if (len(window_shape) > 2):
        raise Exception("Function supports only 2d window")

    shape = a.shape[:-2] + \
            ((a.shape[-2] - window_shape[0]) // dy + 1,) + \
            ((a.shape[-1] - window_shape[1]) // dx + 1,) + \
            (window_shape[0], window_shape[1])  # sausage-like shape with 2D cross-section
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def create_2d_square_rolling_window(a, square_window_size):
    if (a.shape[0] % square_window_size or a.shape[1] % square_window_size):
        raise Exception("""\
            Some elements of the matrix will not get into the rolling window. 
            Expand the original matrix for the following conditions are met:
            a.shape[0] % square_window_size == 0 and a.shape[1] % square_window_size == 0""".strip())

    return rolling_window_2d(a, (square_window_size, square_window_size), square_window_size, square_window_size)


def compare_hist(lpb_patterns_array1, lpb_patterns_array2):
    array1_bins = int(max(lpb_patterns_array1) + 1)
    array2_bins = int(max(lpb_patterns_array2) + 1)
    hist1 = cv.calcHist([np.array(lpb_patterns_array1).astype(np.float32)], [0], None, [array1_bins], [0, array1_bins])
    hist2 = cv.calcHist([np.array(lpb_patterns_array2).astype(np.float32)], [0], None, [array2_bins], [0, array2_bins])
    return cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
