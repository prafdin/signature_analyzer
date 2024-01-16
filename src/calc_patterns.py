import configparser
import pickle
import sqlite3

from skimage import feature
import numpy as np
from PIL import Image


def calc_patterns(connection, image_id):
    cursor = connection.cursor()
    cursor.execute("SELECT image, mode, size FROM prepared_imgs WHERE image_id = ?", (image_id,))
    image_in_bytes, mode, size = cursor.fetchall()[0]
    size = tuple(map(int, size.split(",")))

    image = Image.frombytes(mode=mode, size=size, data=image_in_bytes)
    image = np.array(image)

    LPB_patterns = feature.local_binary_pattern(image, 8, 1, method="uniform").ravel()
    LPB_patterns = list(map(lambda x: int(x), LPB_patterns))

    return LPB_patterns

if __name__ == '__main__':
    RECALCULATE = True

    config = configparser.ConfigParser()
    config.read("configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute('SELECT image_id FROM raw_images WHERE signature_type = 1')
    image_ids = [image_id_row[0] for image_id_row in cursor.fetchall()]

    for image_id in image_ids:
        print(f"Check patterns for image with id = {image_id}")
        cursor.execute('SELECT * FROM images_lbp WHERE image_id = ?', (image_id,))
        if cursor.fetchall() and not RECALCULATE:
            print(f"\tPatterns for image already exist in db")
            continue
        print(f"\tThere is no patterns for image or RECALCULATE option is True. Will calculate")

        lbp_patterns = calc_patterns(connection, image_id)

        lbp_patterns_binary = pickle.dumps(lbp_patterns)

        try:
            cursor.execute("INSERT INTO images_lbp (lbp, image_id) VALUES (?, ?)", (lbp_patterns_binary, image_id))
            print(f"\t\tInserted")
        except sqlite3.Error as er:
            if "UNIQUE constraint failed" in er.args[0]:
                cursor.execute('UPDATE images_lbp SET lbp = ? WHERE image_id = ?', (lbp_patterns_binary, image_id))
                print(f"\t\tUpdated")
            else:
                raise er
        connection.commit()

    connection.close()

