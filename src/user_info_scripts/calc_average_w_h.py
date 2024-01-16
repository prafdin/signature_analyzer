import sqlite3, configparser, cv2
import numpy as np
from PIL import Image, ImageOps

from image_processor.image_processor import ImageProcessor, DefaultImageProcessorSettings

def update_width_high(user_id, images_count):
    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute("""\
        SELECT raw_images.signature_num, images_boxes.bounds FROM raw_images 
        JOIN images_boxes on raw_images.image_id=images_boxes.image_id
        WHERE raw_images.user_id = ? AND raw_images.signature_type = 1
    """, (user_id,))

    images_rows = cursor.fetchall()

    w_array = []
    h_array = []
    for image_row in images_rows[:images_count]:
        bounds = image_row[1]

        min_x, min_y, max_x, max_y = map(int , bounds.split(","))
        image_w = max_x - min_x
        image_h = max_y - min_y
        w_array.append(image_w)
        h_array.append(image_h)

    average_w = int(np.average(w_array))
    average_h = int(np.average(h_array))

    try:
        cursor.execute("INSERT INTO user_info (user_id, signature_width, signature_high) VALUES (?, ?, ?)", (user_id, average_w, average_h))
    except sqlite3.Error as er:
        if "UNIQUE constraint failed" in er.args[0]:
            cursor.execute('UPDATE user_info SET signature_width = ?,  signature_high = ? WHERE user_id = ?', (average_w, average_h, user_id))
        else:
            raise er

    connection.commit()
    connection.close()

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute('SELECT user_id FROM raw_images GROUP BY user_id')
    user_id_rows = cursor.fetchall()
    connection.close()
    user_ids = [user_id_row[0] for user_id_row in user_id_rows]

    images_count = 10

    for user_id in user_ids:
        update_width_high(user_id, images_count)
