import configparser
import sqlite3
import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL import ImageDraw

from image_processor.image_processor import ImageProcessor, DefaultImageProcessorSettings


def find_cog():
    pass


def find_cog_with_counters(image):
    contours, hierarchy = cv2.findContours(image, 1, 2)

    nh, nw = image.shape[:2]

    image = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(image)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 0.3 * nh:
            continue
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # print(f"cx = {cx}; cy = {cy}")
        draw.point([(cx, cy)], (255, 0, 0))

    image.show()

def find_cog_for_full_img(image):
    M = cv2.moments(image)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    image = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(image)

    draw.point([(cx, cy)], (255, 0, 0))

    image.show()

def find_cog_for_sign_only(image, bounds):
    min_x, min_y, max_x, max_y = map(int, bounds.split(","))

    cropped_image = image[min_y:max_y, min_x:max_x]

    M = cv2.moments(cropped_image)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    image = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(image)

    draw.point([(min_x + cx, min_y + cy)], (255, 0, 0))

    image.show()

def calc_img_cog(image):
    M = cv2.moments(image)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return cx, cy

def tests():
    config = configparser.ConfigParser()
    config.read("configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute('SELECT image_path, image_id FROM raw_images WHERE user_id = ? AND signature_type = 1', (1,))
    original_images_rows = cursor.fetchall()
    image_path = original_images_rows[0][0]
    image_id = original_images_rows[0][1]
    cursor.execute("SELECT bounds FROM images_boxes WHERE image_id = ?", (str(image_id),))
    bounds = cursor.fetchall()[0][0]

    image = Image.open(image_path)
    image = ImageProcessor.img_to_gray(image, DefaultImageProcessorSettings)
    image = ImageProcessor.img_to_bin(image, DefaultImageProcessorSettings)

    image = np.array(image.convert('L'))

    # find_cog_for_full_img(image)
    min_x, min_y, max_x, max_y = map(int, bounds.split(","))
    cropped_image = image[min_y:max_y, min_x:max_x]

    find_cog_for_sign_only(image, bounds)

if __name__ == '__main__':
    image_id = "25"

    config = configparser.ConfigParser()
    config.read("configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute('SELECT image_path FROM raw_images WHERE image_id = ?', (image_id,))
    image_row = cursor.fetchall()
    image_path = image_row[0][0]

    cursor.execute("SELECT bounds FROM images_boxes WHERE image_id = ?", (image_id,))
    bounds = cursor.fetchall()[0][0]
    min_x, min_y, max_x, max_y = map(int, bounds.split(","))

    image = Image.open(image_path)

    # TODO: Need to check should we calculate COG for binary image or for original img
    image = ImageProcessor.img_to_gray(image, DefaultImageProcessorSettings)
    image = ImageProcessor.img_to_bin(image, DefaultImageProcessorSettings)

    image = np.array(image.convert('L'))

    image = image[min_y:max_y, min_x:max_x]
    # DEBUG
    # cv2.imshow('Before scratch', image)
    # cv2.waitKey(0)

    cursor.execute(
        "SELECT signature_width, signature_high FROM user_info WHERE user_id = (SELECT user_id FROM raw_images WHERE image_id = ?)",
        (image_id,))
    width, high = cursor.fetchall()[0]
    image = cv2.resize(image, (width, high))
    # DEBUG
    # cv2.imshow('After scratch', image)
    # cv2.waitKey(0)

    cx, cy = calc_img_cog(image)

    # DEBUG
    # image = Image.fromarray(image).convert('RGB')
    # draw = ImageDraw.Draw(image)
    # draw.point([(cx, cy)], (255, 0, 0))

    image.show()