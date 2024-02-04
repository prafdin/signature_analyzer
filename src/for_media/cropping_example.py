import configparser
import sqlite3
from PIL import Image, ImageOps

def show():
    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute("select image_path, bounds from raw_images join images_boxes on raw_images.image_id = images_boxes.image_id")
    image_path, bounds = cursor.fetchall()[0]

    image = Image.open(image_path)
    min_x, min_y, max_x, max_y = map(int, bounds.split(","))

    image.save("../../media/cropping_before.png")

    image = image.crop((min_x, min_y, max_x, max_y))

    image.save("../../media/cropping_after.png")


if __name__ == '__main__':
    show()