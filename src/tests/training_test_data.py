import pickle

import numpy as np

TRAIN_PART = 0.75

np.random.seed(0)

HIST_BINS = 10  # TODO: Check difference with different bins
def get_train_and_tests_lbp_data(connection, max_samples_in_training_part):
    cursor = connection.cursor()
    cursor.execute("""\
        select user_id, count(image_id) from raw_images 
        where raw_images.signature_type = 1
        group by user_id order by user_id
    """)
    user_ids_images_count_rows = cursor.fetchall()

    training_idx = []
    test_idx = []
    for user_id_images_count in user_ids_images_count_rows:
        images_count = user_id_images_count[1]

        user_images_indices = np.random.permutation(images_count)
        count_vectors_in_training_sample = round(len(user_images_indices) * TRAIN_PART)

        if count_vectors_in_training_sample > max_samples_in_training_part:
            count_vectors_in_training_sample = max_samples_in_training_part

        user_training_idx, user_test_idx = user_images_indices[:count_vectors_in_training_sample], user_images_indices[count_vectors_in_training_sample:]
        start_index = len(training_idx) + len(test_idx)
        training_idx += list(np.array(user_training_idx) + start_index)
        test_idx += list(np.array(user_test_idx) + start_index)

    cursor.execute("""\
        select lbp, user_id from images_lbp
        join raw_images on raw_images.image_id = images_lbp.image_id
        where raw_images.signature_type = 1
        ORDER by raw_images.user_id
    """)
    lbp_patterns, user_ids = zip(*cursor.fetchall())
    lbp_patterns = np.array(list(map(lambda obj: pickle.loads(obj), lbp_patterns)))
    user_ids = np.array(list(map(lambda obj: obj, user_ids)))

    histograms = np.array([np.histogram(lbp_pattern, density=True, bins=HIST_BINS, range=(0, HIST_BINS))[0] for lbp_pattern in lbp_patterns])

    np.random.shuffle(training_idx)
    np.random.shuffle(test_idx)

    training, test = histograms[training_idx, :], histograms[test_idx, :]
    training_classes, test_classes = user_ids[training_idx], user_ids[test_idx]

    return training, training_classes.ravel().astype(np.int), test, test_classes.ravel().astype(np.int)

def get_train_and_tests_hog_data(connection, max_samples_in_training_part):
    cursor = connection.cursor()
    cursor.execute("""\
        select user_id, count(image_id) from raw_images 
        where raw_images.signature_type = 1
        group by user_id order by user_id
    """)
    user_ids_images_count_rows = cursor.fetchall()

    training_idx = []
    test_idx = []
    for user_id_images_count in user_ids_images_count_rows:
        images_count = user_id_images_count[1]

        user_images_indices = np.random.permutation(images_count)
        count_vectors_in_training_sample = round(len(user_images_indices) * TRAIN_PART)

        if count_vectors_in_training_sample > max_samples_in_training_part:
            count_vectors_in_training_sample = max_samples_in_training_part

        user_training_idx, user_test_idx = user_images_indices[:count_vectors_in_training_sample], user_images_indices[count_vectors_in_training_sample:]
        start_index = len(training_idx) + len(test_idx)
        training_idx += list(np.array(user_training_idx) + start_index)
        test_idx += list(np.array(user_test_idx) + start_index)

    cursor.execute("""\
        select hog, user_id from images_hog
        join raw_images on raw_images.image_id = images_hog.image_id
        where raw_images.signature_type = 1
        ORDER by raw_images.user_id
    """)
    hog_vectors, user_ids = zip(*cursor.fetchall())
    hog_vectors = np.array(list(map(lambda obj: pickle.loads(obj), hog_vectors)))
    user_ids = np.array(list(map(lambda obj: obj, user_ids)))

    np.random.shuffle(training_idx)
    np.random.shuffle(test_idx)

    training, test = hog_vectors[training_idx, :], hog_vectors[test_idx, :]
    training_classes, test_classes = user_ids[training_idx], user_ids[test_idx]

    return training, training_classes.ravel().astype(np.int), test, test_classes.ravel().astype(np.int)

def get_train_and_tests_direction_data(connection, max_samples_in_training_part):
    cursor = connection.cursor()
    cursor.execute("""\
        select user_id, count(image_id) from raw_images 
        where raw_images.signature_type = 1
        group by user_id order by user_id
    """)
    user_ids_images_count_rows = cursor.fetchall()

    training_idx = []
    test_idx = []
    for user_id_images_count in user_ids_images_count_rows:
        images_count = user_id_images_count[1]

        user_images_indices = np.random.permutation(images_count)
        count_vectors_in_training_sample = round(len(user_images_indices) * TRAIN_PART)

        if count_vectors_in_training_sample > max_samples_in_training_part:
            count_vectors_in_training_sample = max_samples_in_training_part

        user_training_idx, user_test_idx = user_images_indices[:count_vectors_in_training_sample], user_images_indices[count_vectors_in_training_sample:]
        start_index = len(training_idx) + len(test_idx)
        training_idx += list(np.array(user_training_idx) + start_index)
        test_idx += list(np.array(user_test_idx) + start_index)

    cursor.execute("""\
        select direction_v, user_id from images_direction_diagr
        join raw_images on raw_images.image_id = images_direction_diagr.image_id
        where raw_images.signature_type = 1
        ORDER by raw_images.user_id
    """)
    direction_vectors, user_ids = zip(*cursor.fetchall())
    direction_vectors = np.array(list(map(lambda obj: pickle.loads(obj), direction_vectors)))
    user_ids = np.array(list(map(lambda obj: obj, user_ids)))

    np.random.shuffle(training_idx)
    np.random.shuffle(test_idx)

    training, test = direction_vectors[training_idx, :], direction_vectors[test_idx, :]
    training_classes, test_classes = user_ids[training_idx], user_ids[test_idx]

    return training, training_classes.ravel().astype(np.int), test, test_classes.ravel().astype(np.int)