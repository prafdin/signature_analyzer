import sqlite3, configparser

def init_db():
    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]

    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    # signature_num: number of signature in signature class (original or forgery)
    # signature_type: 0 - forgery, 1 - original
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            user_id TEXT NOT NULL,
            signature_num INTEGER,
            signature_type INTEGER,
            UNIQUE(user_id, signature_num, signature_type)
        ) 
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_info (
            user_id INTEGER NOT NULL,
            signature_width INTEGER,
            signature_high INTEGER,
            PRIMARY KEY (user_id)
        ) 
    ''')

    # column 'bounds' contains four numbers separated by commas in next format: x_min,y_min,x_max,y_max
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images_boxes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bounds TEXT NOT NULL, 
            image_id INTEGER ,
            FOREIGN KEY (image_id) REFERENCES raw_images (image_id),
            UNIQUE(image_id)
        ) 
    ''')

    # column 'cog' contains two numbers separated by comma in next format: x,y
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images_cog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cog TEXT NOT NULL, 
            image_id INTEGER ,
            FOREIGN KEY (image_id) REFERENCES raw_images (image_id)
        ) 
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images_lbp (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lbp BLOB NOT NULL, 
            image_id INTEGER ,
            FOREIGN KEY (image_id) REFERENCES raw_images (image_id),
            UNIQUE(image_id)
        ) 
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images_hog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hog BLOB NOT NULL, 
            image_id INTEGER ,
            FOREIGN KEY (image_id) REFERENCES raw_images (image_id),
            UNIQUE(image_id)
        ) 
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images_distance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            distance INTEGER,
            f_image_id INTEGER,
            s_image_id INTEGER,
            UNIQUE(f_image_id, s_image_id)
        ) 
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prepared_imgs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB NOT NULL, 
            image_id INTEGER ,
            mode TEXT NOT NULL, 
            size TEXT NOT NULL, 
            FOREIGN KEY (image_id) REFERENCES raw_images (image_id),
            UNIQUE(image_id)
        ) 
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS thinned_prepared_imgs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB NOT NULL, 
            image_id INTEGER ,
            mode TEXT NOT NULL, 
            size TEXT NOT NULL, 
            FOREIGN KEY (image_id) REFERENCES raw_images (image_id),
            UNIQUE(image_id)
        ) 
    ''')

    connection.commit()
    connection.close()

if __name__ == '__main__':
    init_db()