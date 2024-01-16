import configparser
import sqlite3
import cv2
import numpy as np
from PIL import Image

from image_cog import calc_img_cog
from image_processor.image_processor import ImageProcessor, DefaultImageProcessorSettings
from image_translation import translate


X = 700
Y = 500

def prepare_images(connection, image_id):
    RECALCULATE = False

    print(f"Check image with {image_id} id")
    cursor.execute('SELECT * FROM prepared_imgs WHERE image_id = ?', (image_id,))
    if cursor.fetchall() and not RECALCULATE:
        print(f"\tPrepared image already in db.")
        return
    print(f"\tThere is no prepared image or RECALCULATE option is True. Will prepare image")

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

    image = Image.fromarray(image)

    # раскомментировать для применения скелетирования
    # image = ImageProcessor.thin_img(image, DefaultImageProcessorSettings)

    mode = "1"
    size = ",".join(map(str, image.size))
    image = image.convert(mode)

    image_in_bytes = image.tobytes()

    try:
        cursor.execute("INSERT INTO prepared_imgs (image, image_id, mode, size) VALUES (?, ?, ?, ?)", (image_in_bytes, image_id, "1", size))

    except sqlite3.Error as er:
        if "UNIQUE constraint failed" in er.args[0]:
            cursor.execute('UPDATE prepared_imgs SET image = ?, mode = ?, size = ? WHERE image_id = ?', (image_in_bytes, "1", size, image_id))
        else:
            raise er

    connection.commit()


def tests(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT image, mode, size FROM prepared_imgs WHERE image_id = ?", (565, ))
    image_in_bytes, mode, size = cursor.fetchall()[0]
    size = tuple(map(int, size.split(",")))

    image = Image.frombytes(mode=mode, size=size, data=image_in_bytes)

    image.show()

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    # tests(connection)
    # exit(0)

    cursor.execute('SELECT image_id FROM raw_images WHERE signature_type = 1')
    image_ids = [image_id_row[0] for image_id_row in cursor.fetchall()]

    for image_id in image_ids:
        prepare_images(connection, image_id)

    connection.close()