import configparser
import pickle
import sqlite3

from signature_analyzer.utils import compare_hist


def calc_distance():
    pass

if __name__ == '__main__':
    RECALCULATE = True
    # calc_distance()
    config = configparser.ConfigParser()
    config.read("configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    # Получим айдишники всех оригинальных подписей
    cursor.execute('SELECT image_id FROM raw_images WHERE signature_type = 1')
    image_ids = [image_id_row[0] for image_id_row in cursor.fetchall()]

    # Считаем только для всех фото юзера 1
    cursor.execute("select raw_images.image_id from raw_images where raw_images.user_id = 1 AND raw_images.signature_type = 1")
    user_image_ids = [image_id_row[0] for image_id_row in cursor.fetchall()]

    # Расчитаем расстояние для всех пар айдишников
    # f_image - first image, s_image - second image and etc



    # for f_image_id in image_ids: # Считать для всех пар айпишников
    for f_image_id in user_image_ids: # Считать для выборочных фото с остальными
        # cursor.execute('SELECT lbp FROM images_lbp WHERE image_id = ?', (f_image_id,))
        cursor.execute('SELECT hog FROM images_hog WHERE image_id = ?', (f_image_id,))
        f_lbp = pickle.loads(cursor.fetchall()[0][0])
        for s_image_id in image_ids:
            print(f"Check distance between {f_image_id} and {s_image_id} is in db")
            cursor.execute("select * from images_distance WHERE f_image_id = ? AND s_image_id = ?", (f_image_id, s_image_id))
            if cursor.fetchall() and not RECALCULATE:
                print(f"\tThe distance is in db")
                continue
            print(f"\tThe distance is absent in db or RECALCULATE option is True. Will calculate")

            # cursor.execute('SELECT lbp FROM images_lbp WHERE image_id = ?', (s_image_id,))
            cursor.execute('SELECT hog FROM images_hog WHERE image_id = ?', (s_image_id,))
            s_lbp = pickle.loads(cursor.fetchall()[0][0])

            distance = compare_hist(f_lbp, s_lbp)
            try:
                cursor.execute("INSERT INTO images_distance (f_image_id, s_image_id, distance) VALUES (?, ?, ?)", (f_image_id, s_image_id, distance))
                print(f"\t\tInserted")
            except sqlite3.Error as er:
                if "UNIQUE constraint failed" in er.args[0]:
                    cursor.execute('UPDATE images_distance SET distance = ? WHERE f_image_id = ? AND s_image_id = ?', (distance, f_image_id, s_image_id))
                    print(f"\t\tUpdated")
                else:
                    raise er
            connection.commit()

    connection.close()