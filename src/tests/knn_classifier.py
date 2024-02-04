

# Будем тренировать knn классификатор на 75% всех оригинальных подписей
import configparser
import json
import pickle
import sqlite3
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from output_helper.common import draw_knn_accuracy_plots
from signature_analyzer.classification import knn_classifier_extend_fit, get_knn_classifier, Metric
from tests.training_test_data import get_train_and_tests_lbp_data, get_train_and_tests_hog_data, \
    get_train_and_tests_direction_data

TRAIN_PART = 0.75


def test_classifier(connection):
    k_range = range(2, 19, 2)

    file_data = {}
    scores_EUCL = {}
    for k in k_range:
        training, training_classes, test, test_classes = get_train_and_tests_lbp_data(connection, k)
        knn = get_knn_classifier(k, Metric.EUCLIDEAN)
        knn.fit(training, training_classes.reshape(-1, 1))
        Y_pred_EUCL = knn.predict(test).astype(int)
        scores_EUCL[k] = metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_EUCL)
        file_data[k] = {
            "accuracy": metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_EUCL),
            "f1": metrics.f1_score(test_classes.ravel().astype(np.int), Y_pred_EUCL, average='macro')
        }

    scores_BHAT = {}
    for k in k_range:
        training, training_classes, test, test_classes = get_train_and_tests_lbp_data(connection, k)
        knn = get_knn_classifier(k, Metric.BHATTACHARYYA)
        knn.fit(training, training_classes.reshape(-1, 1))
        Y_pred_BHAT = knn.predict(test).astype(int)
        scores_BHAT[k] = metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_BHAT)
        file_data[k] = {
            "accuracy": metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_BHAT),
            "f1": metrics.f1_score(test_classes.ravel().astype(np.int), Y_pred_BHAT, average='micro')
        }

    scores_KVDIV = {}
    for k in k_range:
        training, training_classes, test, test_classes = get_train_and_tests_lbp_data(connection, k)
        knn = get_knn_classifier(k, Metric.KV_DIVERGENCE)
        knn.fit(training, training_classes.reshape(-1, 1))
        Y_pred_KVDIV = knn.predict(test).astype(int)
        scores_KVDIV[k] = metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_KVDIV)
        file_data[k] = {
            "accuracy": metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_KVDIV),
            "f1": metrics.f1_score(test_classes.ravel().astype(np.int), Y_pred_KVDIV, average='micro')
        }

    # scores_COREL = {}
    # for k in k_range:
    #     knn = get_knn_classifier(k, Metric.CORRELATION)
    #     knn.fit(training, training_classes.reshape(-1, 1))
    #     Y_pred_COREL = knn.predict(test).astype(int)
    #     scores_COREL[k] = metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_COREL)


    with open("knn_stats.json", "w") as f:
        json.dump(file_data, f)

    draw_knn_accuracy_plots(k_range, scores_EUCL.values(), scores_BHAT.values(), scores_KVDIV.values())

    plt.show()


def another_test_classifier(training, training_classes, test, test_classes):
    k = 5
    knn = get_knn_classifier(k, Metric.BHATTACHARYYA)
    knn.fit(training, training_classes.reshape(-1, 1))
    Y_pred_BHAT = knn.predict(test).astype(int)
    accuracy = metrics.accuracy_score(test_classes.ravel().astype(np.int), Y_pred_BHAT)
    f1 = metrics.f1_score(test_classes.ravel().astype(np.int), Y_pred_BHAT, average='macro')
    print(f"Accuracy: {accuracy}, F1: {f1}")

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
    # print("Тренеровочная и тестовые выборки сформированы")

    # training, training_classes, test, test_classes = get_train_and_tests_lbp_data(connection, 18)
    # training, training_classes, test, test_classes = get_train_and_tests_hog_data(connection, 18)
    training, training_classes, test, test_classes = get_train_and_tests_direction_data(connection, 18)
    another_test_classifier(training, training_classes, test, test_classes)
    # test_classifier(connection)

    # print(training_classes)
    # train_classifier()

    connection.close()