import sqlite3, configparser, cv2
import numpy as np
from PIL import Image, ImageOps

from image_processor.image_processor import ImageProcessor, DefaultImageProcessorSettings

def calc_bounds(image_path):
    image = Image.open(image_path)
    image = ImageProcessor.img_to_gray(image, DefaultImageProcessorSettings)
    image = ImageProcessor.img_to_bin(image, DefaultImageProcessorSettings)

    image = np.array(image.convert('L'))

    kernel = np.ones((15, 15), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

    cnts = cv2.findContours(image, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    # DEBUG
    # for i in cnts:
    #     x, y, w, h = cv2.boundingRect(i)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 4)
    #
    # cv2.imshow('Contours', image)
    # cv2.waitKey(0)
    #######

    nh, nw = image.shape[:2]
    min_x, min_y, max_x, max_y = 999999999, 999999999, -1, -1
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= 0.1 * nh:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x + w > max_x:
                max_x = x + w
            if y + h > max_y:
                max_y = y + h

    return min_x, min_y, max_x, max_y

def update_width_high(user_id):
    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute('SELECT image_path, image_id FROM raw_images WHERE user_id = ?', (user_id,))
    original_images_rows = cursor.fetchall()

    for orig_image_row in original_images_rows:
        orig_image_path = orig_image_row[0]
        image_id = orig_image_row[1]
        min_x, min_y, max_x, max_y = calc_bounds(orig_image_path)

        try:
            cursor.execute("INSERT INTO images_boxes (bounds, image_id) VALUES (?, ?)", (",".join(map(str, [min_x, min_y, max_x, max_y])), image_id) )
        except sqlite3.Error as er:
            if "UNIQUE constraint failed" in er.args[0]:
                cursor.execute('UPDATE images_boxes SET bounds = ? WHERE image_id = ?', (",".join(map(str, [min_x, min_y, max_x, max_y])), image_id))
            else:
                raise er


    connection.commit()
    connection.close()

def test():
    update_width_high(52)

if __name__ == '__main__':
    # test()
    # exit(0)

    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute('SELECT user_id FROM raw_images GROUP BY user_id')
    user_id_rows = cursor.fetchall()
    connection.close()
    user_ids = [user_id_row[0] for user_id_row in user_id_rows]

    for user_id in user_ids:
        update_width_high(user_id)
