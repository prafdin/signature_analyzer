import math

import PIL
import numpy as np
from PIL import ImageFilter, ImageEnhance
from PIL.Image import Image
import cv2 as cv

from image_processor.zhang_suen import zhangSuen
from output_helper.image_output import image_output


class ImageProcessorSettings:
    bin_threshold = -1
    morph_open_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2)) # MORPH_ELLIPSE
    contrast_factor = 20
    dilate_iterations = 1
    erode_iterations = 1


DefaultImageProcessorSettings = ImageProcessorSettings()


class ImageProcessor:
    @staticmethod
    @image_output
    def blur_img(image: Image, settings: ImageProcessorSettings) -> Image:
        return image.filter(filter=ImageFilter.BLUR)

    @staticmethod
    @image_output
    def img_to_gray(image: Image, settings: ImageProcessorSettings) -> Image:
        image_copy = image.copy()
        return image_copy.convert("L")

    @staticmethod
    @image_output
    def img_to_bin(image: Image, settings: ImageProcessorSettings) -> Image:
        threshold = settings.bin_threshold
        if threshold < 0:
            _, bin_img = cv.threshold(np.array(image, dtype=np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # bin_img = cv.adaptiveThreshold(np.array(image, dtype=np.uint8), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
            bin_img = np.invert(bin_img)
            return PIL.Image.fromarray(bin_img).convert("1")
        else:
            return image.point(lambda x: 255 * (x < threshold), mode='1')

    @staticmethod
    @image_output
    def thin_img(image: Image, settings: ImageProcessorSettings) -> Image:
        image = image.convert("1")
        thinned = zhangSuen(np.array(image, dtype=np.uint8)).astype(int)
        return PIL.Image.fromarray(thinned.astype(np.uint8) * 255, "L")

    @staticmethod
    @image_output
    def morph_open(image: Image, settings: ImageProcessorSettings) -> Image:
        kernel = settings.morph_open_kernel
        img = np.array(image).astype(np.uint8)
        img = cv.morphologyEx(np.array(image).astype(np.uint8), cv.MORPH_OPEN, kernel)
        return PIL.Image.fromarray(img.astype(np.uint8) * 255, "L")

    @staticmethod
    @image_output
    def dilate_img(image: Image, settings: ImageProcessorSettings) -> Image:
        kernel = settings.morph_open_kernel
        img = np.array(image).astype(np.uint8)

        dilate_iterations = settings.dilate_iterations
        img = cv.dilate(img, kernel, iterations=dilate_iterations)
        return PIL.Image.fromarray(img.astype(np.uint8) * 255, "L")

    @staticmethod
    @image_output
    def erode_img(image: Image, settings: ImageProcessorSettings) -> Image:
        kernel = settings.morph_open_kernel
        img = np.array(image).astype(np.uint8)

        erode_iterations = settings.erode_iterations
        img = cv.erode(img, kernel, iterations=erode_iterations)
        return PIL.Image.fromarray(img.astype(np.uint8) * 255, "L")

    @staticmethod
    @image_output
    def contrast_img(image: Image, settings: ImageProcessorSettings) -> Image:
        contrast_factor = settings.contrast_factor
        return ImageEnhance.Contrast(image).enhance(contrast_factor)