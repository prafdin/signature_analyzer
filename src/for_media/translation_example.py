import configparser
import sqlite3

import cv2
import numpy as np
from PIL import Image, ImageOps

from image_cog import calc_img_cog
from image_processor.image_processor import ImageProcessor, DefaultImageProcessorSettings
from image_translation import translate


def show():
    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute("select image_path, bounds, signature_width, signature_high from raw_images join images_boxes on raw_images.image_id = images_boxes.image_id join user_info on raw_images.user_id = user_info.user_id")
    image_path, bounds, width, high = cursor.fetchall()[0]

    X = 700
    Y = 500

    min_x, min_y, max_x, max_y = map(int, bounds.split(","))

    image = Image.open(image_path)

    image = ImageProcessor.img_to_gray(image, DefaultImageProcessorSettings)
    # image = ImageProcessor.contrast_img(image, DefaultImageProcessorSettings)
    image = ImageProcessor.img_to_bin(image, DefaultImageProcessorSettings)
    image = ImageProcessor.morph_open(image, DefaultImageProcessorSettings)

    image = ImageProcessor.dilate_img(image, DefaultImageProcessorSettings)
    image = ImageProcessor.erode_img(image, DefaultImageProcessorSettings)

    image = np.array(image.convert('L'))

    image = image[min_y:max_y, min_x:max_x]
    image = cv2.resize(image, (width, high))
    cx, cy = calc_img_cog(image)

    image = Image.open(image_path)
    image = np.array(image.convert('L'))
    image = image[min_y:max_y, min_x:max_x]
    image = cv2.resize(image, (width, high))

    cv2.imwrite('../../media/img_before_translation.png', image)
    image = translate(image, X, Y, cx, cy)
    cv2.imwrite('../../media/img_after_translation.png', image)

    cv2.imshow('Translation', image)
    cv2.waitKey()


if __name__ == '__main__':
    show()