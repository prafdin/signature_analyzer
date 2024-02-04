import configparser
import pickle
import sqlite3

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def calc_direction(connection, image_id):
    cursor = connection.cursor()
    cursor.execute("SELECT image, mode, size FROM thinned_prepared_imgs WHERE image_id = ?", (image_id,))
    image_in_bytes, mode, size = cursor.fetchall()[0]
    size = tuple(map(int, size.split(",")))

    image = Image.frombytes(mode=mode, size=size, data=image_in_bytes)
    image = np.array(image)
    return calc_patterns(image)


def get_shifted_window(data, n_c_x, n_c_y, w_size):
    shifted_window = np.zeros((w_size, w_size))

    c_x, c_y = 1, 1
    offset_x = n_c_x - c_x
    offset_y = n_c_y - c_y

    for i in range(w_size):
        offset_i = i + offset_x
        if offset_i < 0 or offset_i > data.shape[0] - 1:
            continue
        for j in range(w_size):
            offset_j = j + offset_y
            if offset_j < 0 or offset_j > data.shape[1] - 1:
                continue
            shifted_window[i][j] = data[offset_i][offset_j]
    return shifted_window

def calc_patterns(img):
    w_size = 3

    indices = np.argwhere(np.apply_along_axis(lambda x: x == 1, axis=0, arr=img))
    window_array = [get_shifted_window(img, *coords, w_size) for coords in indices]

    feature_v = [w.ravel().astype(np.int) for w in window_array]
    feature_v = [int("".join(map(str, v)), 2) for v in feature_v]

    bins = 2 ** (w_size * w_size)
    feature_v, bins = np.histogram(feature_v, density=True, bins=bins, range=(0, bins))

    # DEBUG
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.show()

    return feature_v

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
        print(f"Check distance_v for image with id = {image_id}")
        cursor.execute('SELECT * FROM images_direction_diagr WHERE image_id = ?', (image_id,))
        if cursor.fetchall() and not RECALCULATE:
            print(f"\tdistance_v for image already exist in db")
            continue
        print(f"\tThere is no distance_v for image or RECALCULATE option is True. Will calculate")

        direction_v = calc_direction(connection, image_id)
        direction_v_binary = pickle.dumps(direction_v)

        try:
            cursor.execute("INSERT INTO images_direction_diagr (direction_v, image_id) VALUES (?, ?)", (direction_v_binary, image_id))
            print(f"\t\tInserted")
        except sqlite3.Error as er:
            if "UNIQUE constraint failed" in er.args[0]:
                cursor.execute('UPDATE images_direction_diagr SET direction_v = ? WHERE image_id = ?', (direction_v_binary, image_id))
                print(f"\t\tUpdated")
            else:
                raise er
        connection.commit()

    connection.close()