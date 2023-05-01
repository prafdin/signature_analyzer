import PIL
from PIL.Image import Image

from matplotlib import pyplot as plt, gridspec

from output_helper import config


def image_output(img_process_func):
    if 'image_output' in config \
            and img_process_func.__name__ in config['image_output'] \
            and config['image_output'][img_process_func.__name__]:
        def process_image_with_output(image: Image, settings):
            processed_img = img_process_func(image, settings)
            # processed_img.show()
            draw_plots(img_process_func.__name__, image, processed_img)
            return processed_img
    else:
        def process_image_with_output(image: PIL.Image.Image, settings):
            return img_process_func(image, settings)
    return process_image_with_output


def draw_plots(title, img_before, img_after):
    fig = plt.figure(num=title, figsize=(10, 10))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5, wspace=1)
    gs = gridspec.GridSpec(1, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_title('Original image')
    ax00.imshow(img_before, cmap='gray', vmin=0, vmax=255)

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_title('Processed image')
    ax01.imshow(img_after, cmap='gray', vmin=0, vmax=255)
