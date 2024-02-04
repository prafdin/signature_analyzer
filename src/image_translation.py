import configparser
import sqlite3, cv2

import numpy as np
from PIL import Image

from image_cog import calc_img_cog
from image_processor.image_processor import ImageProcessor, DefaultImageProcessorSettings


def translate(image, X, Y, cx, cy):
    x0 = X // 2
    y0 = Y // 2
    # x_c, y_c = img.shape[1], img.shape[0]
    # x_c, y_c = x_c // 2, y_c // 2
    # x_c, y_c = 0, 0
    translation_matrix = np.float32([[1, 0, x0 - cx], [0, 1, y0 - cy]])
    img_translation = cv2.warpAffine(image, translation_matrix, (X, Y), borderValue=(255, 255, 255))

    #DEBUG
    # cv2.imshow('Translation', img_translation)
    # cv2.waitKey()
    return img_translation

if __name__ == '__main__':
    X = 700
    Y = 500

    image_id = "25"

    config = configparser.ConfigParser()
    config.read("configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute('SELECT image_path FROM raw_images WHERE image_id = ?', (image_id,))
    image_row = cursor.fetchall()
    image_path = image_row[0][0]
    img = cv2.imread(image_path)

    cursor.execute("SELECT bounds FROM images_boxes WHERE image_id = ?", (image_id,))
    bounds = cursor.fetchall()[0][0]
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

    cursor.execute(
        "SELECT signature_width, signature_high FROM user_info WHERE user_id = (SELECT user_id FROM raw_images WHERE image_id = ?)",
        (image_id,))
    width, high = cursor.fetchall()[0]
    image = cv2.resize(image, (width, high))

    cx, cy = calc_img_cog(image)

    image = translate(image, X, Y, cx, cy)

    # image = Image.fromarray(image)
    # image = ImageProcessor.thin_img(image, DefaultImageProcessorSettings)
    # image = np.array(image)

    # cv2.imshow("test", image)
    # cv2.waitKey(0)