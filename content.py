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
    print(y)
    matrix_order = np.argsort(Dist, kind='mergesort')
    return y[matrix_order]


def p_y_x_knn(y, k):
    k_nearest = y[:,:k] - 1
    occr_matrix = np.stack(np.bincount(k_nearest[i], minlength=4) for i in range(k_nearest.shape[0]))
    return occr_matrix / k


def classification_error(p_y_x, y_true):
    print(p_y_x)
    prediction = p_y_x.shape[1] - np.argmax(p_y_x[:,::-1], axis=1)
    diff = prediction - y_true
    return np.count_nonzero(diff) / diff.shape[0]


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    train_labels = sort_train_labels_knn(hamming_distance(Xval, Xtrain), ytrain)
    err = list(map(lambda k: classification_error(p_y_x_knn(train_labels, k), yval), k_values))
    min_index = np.argmin(err)
    pass
    #return err[min_index], k_values[min_index], err


def estimate_a_priori_nb(ytrain):
    return np.bincount(ytrain)[1:] / ytrain.shape[0]


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    y_eq_k = np.stack(map(lambda k: np.equal(ytrain, k), range(1, 5)))
    result = np.stack(map(lambda r: np.sum(np.bitwise_and(Xtrain.toarray().T, r), axis=1), y_eq_k))
    return ((result + a - 1).T / (np.sum(y_eq_k, axis=1) + a + b - 2)).T

def p_y_x_nb(p_y, p_x_1_y, X):
    X = X.toarray()
    M = p_x_1_y.shape[0]
    N = X.shape[0]
    p_x_l_y_neg = np.ones(p_x_1_y.shape) - p_x_1_y

    result = np.zeros((N, M))

    for n in range(N):
        for k in range(M):
            result[n, k] = p_y[k] * np.prod(np.where(X[n], p_x_1_y[k], p_x_l_y_neg[k]))

        result[n, :] /= np.sum(result[n, :])

    return result


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    errors = np.zeros((len(a_values), len(b_values)))
    p_y = estimate_a_priori_nb(ytrain)

    for a_index in range(len(a_values)):
        for b_index in range(len(b_values)):
            p_x_y = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a_index], b_values[b_index])
            p_y_x = p_y_x_nb(p_y, p_x_y, Xval)
            errors[a_index, b_index] = classification_error(p_y_x, yval)

    index = np.unravel_index(errors.argmin(), errors.shape)

    best_error = errors[index]
    best_a = a_values[index[0]]
    best_b = b_values[index[1]]

    return best_error, best_a, best_b, errors

"""
def p_y_x_nb(p_y, p_x_1_y, X):

    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.

    pass


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):

    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)

    pass
"""
