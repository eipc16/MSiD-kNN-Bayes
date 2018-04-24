# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np

def hamming_distance(X, X_train):
    A, A_trainT = X.toarray(), np.transpose(X_train).toarray()
    return A@(1 - A_trainT) + (1 - A)@A_trainT

def sort_train_labels_knn(Dist, y):
    matrix_order = np.argsort(Dist, kind='mergesort')
    return y[matrix_order]

def p_y_x_knn(y, k):
    k_nearest = y[:,:k]
    occr_matrix = np.stack(np.bincount(k_nearest[i], minlength=4) for i in range(k_nearest.shape[0]))
    return occr_matrix / k

def classification_error(p_y_x, y_true):
    prediction = (p_y_x.shape[1] - 1) - np.argmax(p_y_x[:,::-1], axis=1)
    diff = prediction - y_true
    return np.count_nonzero(diff) / diff.shape[0]

def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    train_labels = sort_train_labels_knn(hamming_distance(Xval, Xtrain), ytrain)
    err = list(map(lambda k: classification_error(p_y_x_knn(train_labels, k), yval), k_values))
    min_index = np.argmin(err)
    return err[min_index], k_values[min_index], err

def estimate_a_priori_nb(ytrain):
    return np.bincount(ytrain) / ytrain.shape[0]

def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    y_eq_k = np.stack(map(lambda k: np.equal(ytrain, k), range(4)))
    result = np.stack(map(lambda r: np.sum(np.bitwise_and(Xtrain.toarray().T, r), axis=1), y_eq_k))
    return ((result + a - 1).T / (np.sum(y_eq_k, axis=1) + a + b - 2)).T

#wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas
def p_y_x_nb(p_y, p_x_1_y, X):
    N, M = X.shape[0], p_y.shape[0]
    X = X.toarray()

    def f(n, m):
        return np.prod(np.logical_not(X[n, :]) - p_x_1_y[m, :])

    result = np.fromfunction(np.vectorize(f), shape=(N, M), dtype=int) * p_y
    result /= result @ np.ones(shape=(4, 1))

    return result

def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    A, B = len(a_values), len(b_values)
    p_y = estimate_a_priori_nb(ytrain)

    def get_error(a, b):
        p_x_y_nb = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a], b_values[b])
        p_y_x = p_y_x_nb(p_y, p_x_y_nb, Xval)
        return classification_error(p_y_x, yval)

    errors = np.fromfunction(np.vectorize(get_error), shape=(A, B), dtype=int)

    min = np.argmin(errors)
    minA, minB = min // A, min % A
    return (errors[minA, minB], a_values[minA], b_values[minB], errors)
