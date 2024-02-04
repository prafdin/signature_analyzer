import configparser
import sqlite3

import numpy as np
from sklearn import metrics
from sklearn import svm
# from sklearn.svm import LinearSVC

from tests.training_test_data import get_train_and_tests_lbp_data, get_train_and_tests_hog_data, \
    get_train_and_tests_direction_data


def test_classifier(training, training_classes, test, test_classes):
    clf = svm.SVC(decision_function_shape='ovo', kernel='rbf')
    clf.fit(training, training_classes)
    predict_test_classes = clf.predict(test)

    accuracy = metrics.accuracy_score(test_classes.ravel().astype(np.int), predict_test_classes)
    f1 = metrics.f1_score(test_classes.ravel().astype(np.int), predict_test_classes, average='macro')
    print(f"Accuracy: {accuracy}, F1: {f1}")

    # print(metrics.classification_report(test_classes, predict_test_classes))

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("../configs.ini")

    database_name = config["DB"]["path"]
    connection = sqlite3.connect(database_name)

    # tests(connection)
    # exit(0)

    # training, training_classes, test, test_classes = get_train_and_tests_lbp_data(connection, 18)
    # training, training_classes, test, test_classes = get_train_and_tests_hog_data(connection, 18)
    training, training_classes, test, test_classes = get_train_and_tests_direction_data(connection, 5)

    test_classifier(training, training_classes, test, test_classes)

    connection.close()