

# Будем тренировать knn классификатор на 75% всех оригинальных подписей
import configparser
import pickle
import sqlite3
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from output_helper.common import draw_knn_accuracy_plots
from signature_analyzer.classification import knn_classifier_extend_fit, get_knn_classifier, Metric

TRAIN_PART = 0.75

HIST_BINS = 10  # TODO: Check difference with different bins
def get_train_and_tests_lbp_data(connection):
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

    training, test = histograms[training_idx, :], histograms[test_idx, :]
    training_classes, test_classes = user_ids[training_idx], user_ids[test_idx]

    return training, training_classes, test, test_classes


def get_train_and_tests_hog_data(connection):
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

    training, test = hog_vectors[training_idx, :], hog_vectors[test_idx, :]
    training_classes, test_classes = user_ids[training_idx], user_ids[test_idx]

    return training, training_classes, test, test_classes

def test_classifier(training, training_classes, test, test_classes):
    k_range = range(1, 15)

    scores_EUCL = {}
    for k in k_range:
        knn = get_knn_classifier(k, Metric.EUCLIDEAN)
        knn.fit(training, training_classes.reshape(-1, 1))
        Y_pred_EUCL = knn.predict(test).astype(int)
        scores_EUCL[k] = metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_EUCL)

    scores_BHAT = {}
    for k in k_range:
        knn = get_knn_classifier(k, Metric.BHATTACHARYYA)
        knn.fit(training, training_classes.reshape(-1, 1))
        Y_pred_BHAT = knn.predict(test).astype(int)
        scores_BHAT[k] = metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_BHAT)

    scores_KVDIV = {}
    for k in k_range:
        knn = get_knn_classifier(k, Metric.KV_DIVERGENCE)
        knn.fit(training, training_classes.reshape(-1, 1))
        Y_pred_KVDIV = knn.predict(test).astype(int)
        scores_KVDIV[k] = metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_KVDIV)

    scores_COREL = {}
    for k in k_range:
        knn = get_knn_classifier(k, Metric.CORRELATION)
        knn.fit(training, training_classes.reshape(-1, 1))
        Y_pred_COREL = knn.predict(test).astype(int)
        scores_COREL[k] = metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_COREL)

    draw_knn_accuracy_plots(k_range, scores_EUCL.values(), scores_BHAT.values(), scores_KVDIV.values(),
                            scores_COREL.values())

    plt.show()


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def tests(connection):
    cursor = connection.cursor()
    cursor.execute("""\
        select lbp from images_lbp
    """)
    lbp_patterns = cursor.fetchall()[:3]
    lbp_patterns = np.array(list(map(lambda obj: pickle.loads(obj[0]), lbp_patterns)))

    some_pattern = lbp_patterns[0]
    another_pattern = lbp_patterns[1]

    n_bins1 = int(some_pattern.max() + 1)
    hist1, _ = np.histogram(some_pattern, density=True, bins=n_bins1, range=(0, n_bins1))

    n_bins2 = int(some_pattern.max() + 1)
    hist2, _ = np.histogram(another_pattern, density=True, bins=n_bins2, range=(0, n_bins2))

    score = kullback_leibler_divergence(hist1, hist2)
    print(f"score = {score}, n_bins1 = {n_bins1}, n_bins2 = {n_bins2}")

    # fig, axs = plt.subplots(1, 1, tight_layout=True)
    # axs.hist(some_pattern, bins=n_bins, density=True)
    # plt.show()

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)

    # tests(connection)
    # exit(0)

    # training, training_classes, test, test_classes = get_train_and_tests_lbp_data(connection)
    training, training_classes, test, test_classes = get_train_and_tests_hog_data(connection)
    print("Тренеровочная и тестовые выборки сформированы")


    test_classifier(training, training_classes, test, test_classes)

    # print(training_classes)
    # train_classifier()

    connection.close()