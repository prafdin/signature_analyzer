import sqlite3, configparser, os, re


CEDAR_PATH = r"C:\Users\ivpr0122\Documents\Personal\signature_analyzer\CEDAR"

def upload_images():
    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]

    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    users_dirs = os.listdir(CEDAR_PATH)
    for user_id_str in users_dirs:
        user_dir = f"{CEDAR_PATH}\\{user_id_str}"
        user_images = os.listdir(user_dir)
        for user_image_name in user_images:
            user_image_path = f"{user_dir}\\{user_image_name}"

            signature_type = int("original" in user_image_name)
            user_id = int(user_id_str)
            signature_num = re.compile(r"(?<=_)\d*(?=\.)").search(user_image_path).group(0)

            cursor.execute(
                'INSERT INTO raw_images (image_path, user_id, signature_num, signature_type) VALUES (?, ?, ?, ?)',
                (user_image_path, user_id, signature_num, signature_type)
            )

    connection.commit()
    connection.close()


if __name__ == '__main__':
    upload_images()