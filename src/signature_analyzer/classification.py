from enum import Enum
import cv2 as cv
import numpy as np
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import distance_metrics

def ad(x, y ):
    d = distance_metrics()['euclidean']
    return d

class Metric(Enum):
    EUCLIDEAN = lambda x, y: distance_metrics()['euclidean'](x.reshape(1, -1), y.reshape(1, -1))
    CITYBLOCK = lambda x, y: distance_metrics()['cityblock'](x.reshape(1, -1), y.reshape(1, -1))
    MANHATTAN = lambda x, y: distance_metrics()['manhattan'](x.reshape(1, -1), y.reshape(1, -1))
    BHATTACHARYYA = lambda x, y: cv.compareHist(x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32), cv.HISTCMP_BHATTACHARYYA)
    CORRELATION = lambda x, y: cv.compareHist(x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32), cv.HISTCMP_CORREL)
    KV_DIVERGENCE = lambda x, y: cv.compareHist(x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32), cv.HISTCMP_KL_DIV)


def get_knn_classifier(k, metric: Metric):
    return KNeighborsClassifier(n_neighbors=k, metric=metric)


def knn_classifier_extend_fit(knn, X, Y):
    """
    Returned fitted classifier with prediction on training set.
    """
    knn = sklearn.clone(knn)

    Y_pred = []
    for idx in range(len(X)):
        x_i = X.take(idx, axis=0).reshape(1, -1)
        X_i = np.delete(X, idx, axis=0)
        Y_i = np.delete(Y, idx)
        knn.fit(X_i, Y_i)
        Y_pred.append(int(knn.predict(x_i)))

    knn.fit(X, Y.ravel())
    return knn, Y_pred
