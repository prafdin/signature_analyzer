import PIL
import numpy as np
from PIL import ImageFilter
from PIL.Image import Image

from image_processor.zhang_suen import zhangSuen
from output_helper.image_output import image_output


class ImageProcessorSettings:
    bin_threshold = 125


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
        return image.point(lambda x: 255 * (x > threshold), mode='1')

    @staticmethod
    @image_output
    def thin_img(image: Image, settings: ImageProcessorSettings) -> Image:
        thinned = zhangSuen(np.invert(image)).astype(int)
        return PIL.Image.fromarray(thinned.astype(np.uint8) * 255, "L")
