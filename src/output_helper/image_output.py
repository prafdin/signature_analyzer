import PIL
from PIL.Image import Image

# from image_processor.image_processor import ImageProcessorSettings
from output_helper import config


def image_output(img_process_func):
    if 'image_output' in config \
            and img_process_func.__name__ in config['image_output'] \
            and config['image_output'][img_process_func.__name__]:
        def process_image_with_output(image: Image, settings):
            processed_img = img_process_func(image, settings)
            processed_img.show()
            return processed_img
    else:
        def process_image_with_output(image: PIL.Image.Image, settings):
            return img_process_func(image, settings)
    return process_image_with_output
