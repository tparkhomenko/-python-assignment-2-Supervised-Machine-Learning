import numpy as np


# TODO: ask about code style here
# noinspection PyPep8Naming
class kNN:

    # TODO: code __init__
    def __init__(self, n_neighbors):
        self._n_neighbors = n_neighbors

    # TODO: code fit
    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    # TODO: code predict
    def predict(self, x_test):
        predicts = np.zeros([len(x_test)], dtype=int)
        for n in range(len(x_test)):
            predicts[n] = self._point_predict(x_test[n])
        return predicts

    @staticmethod
    def _distance(a_point, b_point):  # Euclidean distance, static method
        return np.linalg.norm(a_point - b_point)

    @staticmethod
    def _onepoint(x_testpoint, X_train):
        dists = np.zeros([len(X_train)])
        for n in range(len(X_train)):
            dists[n] = kNN._distance(x_testpoint, X_train[n])
        return dists.argsort()  # points indexes of sorted distances

    def _point_predict(self, x_testpoint): # return which feature, when tey are equal?
        order = kNN._onepoint(x_testpoint, self._X_train)[0:self._n_neighbors]
        nearest_k_y = self._y_train[order]
        return np.argmax(np.bincount(nearest_k_y))


