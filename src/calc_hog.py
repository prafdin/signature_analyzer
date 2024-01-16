import configparser
import pickle
import sqlite3

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.metrics.pairwise import distance_metrics

from image_cog import calc_img_cog
from image_processor.image_processor import ImageProcessor, DefaultImageProcessorSettings
from image_translation import translate
from signature_analyzer.classification import Metric

X = 700
Y = 500


def main_flow():
    f_image_path = r"C:\Users\ivpr0122\Documents\Personal\signature_analyzer\CEDAR\1\original_1_1.png"
    s_image_path = r"C:\Users\ivpr0122\Documents\Personal\signature_analyzer\CEDAR\4\original_4_1.png"



    # FIRST
    f_image = Image.open(f_image_path)
    f_bounds = "43,64,576,309"
    f_min_x, f_min_y, f_max_x, f_max_y = map(int, f_bounds.split(","))

    f_image = ImageProcessor.img_to_gray(f_image, DefaultImageProcessorSettings)
    # image = ImageProcessor.contrast_img(image, DefaultImageProcessorSettings)
    f_image = ImageProcessor.img_to_bin(f_image, DefaultImageProcessorSettings)
    f_image = ImageProcessor.morph_open(f_image, DefaultImageProcessorSettings)

    f_image = ImageProcessor.dilate_img(f_image, DefaultImageProcessorSettings)
    f_image = ImageProcessor.erode_img(f_image, DefaultImageProcessorSettings)

    f_image = np.array(f_image.convert('L'))
    f_image = f_image[f_min_x:f_max_y, f_min_x:f_max_x]
    f_cx, f_cy = calc_img_cog(f_image)

    f_image = translate(f_image, X, Y, f_cx, f_cy)

    f_hog_vector = hog(f_image, orientations=8, pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1), feature_vector=True)
    ##########################

    # Second
    s_image = Image.open(s_image_path)
    s_bounds = "10,68,395,176"
    s_min_x, s_min_y, s_max_x, s_max_y = map(int, s_bounds.split(","))

    s_image = ImageProcessor.img_to_gray(s_image, DefaultImageProcessorSettings)
    # image = ImageProcessor.contrast_img(image, DefaultImageProcessorSettings)
    s_image = ImageProcessor.img_to_bin(s_image, DefaultImageProcessorSettings)
    s_image = ImageProcessor.morph_open(s_image, DefaultImageProcessorSettings)

    s_image = ImageProcessor.dilate_img(s_image, DefaultImageProcessorSettings)
    s_image = ImageProcessor.erode_img(s_image, DefaultImageProcessorSettings)

    s_image = np.array(s_image.convert('L'))
    s_image = s_image[s_min_x:s_max_y, s_min_x:s_max_x]
    s_cx, s_cy = calc_img_cog(s_image)

    s_image = translate(s_image, X, Y, s_cx, s_cy)

    s_hog_vector = hog(s_image, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), feature_vector=True)
    ##########################

    d = Metric.EUCLIDEAN(f_hog_vector, s_hog_vector)
    print(d)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    #
    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    #
    # # Rescale histogram for better display
    # # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    #
    # ax2.axis('off')
    # ax2.imshow(hog_image, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

    # image.show()

def calc_hog(connection, image_id):
    cursor = connection.cursor()
    cursor.execute("SELECT image, mode, size FROM prepared_imgs WHERE image_id = ?", (image_id,))
    image_in_bytes, mode, size = cursor.fetchall()[0]
    size = tuple(map(int, size.split(",")))

    image = Image.frombytes(mode=mode, size=size, data=image_in_bytes)
    image = np.array(image.convert('L'))

    hog_v = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)

    return hog_v

if __name__ == '__main__':
    # main_flow()
    # exit(0)
    RECALCULATE = True

    config = configparser.ConfigParser()
    config.read("configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute('SELECT image_id FROM raw_images WHERE signature_type = 1')
    image_ids = [image_id_row[0] for image_id_row in cursor.fetchall()]

    for image_id in image_ids:
        print(f"Check hog for image with id = {image_id}")
        cursor.execute('SELECT * FROM images_lbp WHERE image_id = ?', (image_id,))
        if cursor.fetchall() and not RECALCULATE:
            print(f"\tHog v for image already exist in db")
            continue
        print(f"\tThere is no hog v for image or RECALCULATE option is True. Will calculate")

        hog_v = calc_hog(connection, image_id)

        hog_v_binary = pickle.dumps(hog_v)

        try:
            cursor.execute("INSERT INTO images_hog (hog, image_id) VALUES (?, ?)", (hog_v_binary, image_id))
            print(f"\t\tInserted")
        except sqlite3.Error as er:
            if "UNIQUE constraint failed" in er.args[0]:
                cursor.execute('UPDATE images_hog SET hog = ? WHERE image_id = ?', (hog_v_binary, image_id))
                print(f"\t\tUpdated")
            else:
                raise er
        connection.commit()

    connection.close()