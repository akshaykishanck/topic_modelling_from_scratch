import numpy as np
from topic_modelling.utils.utils import sigmoid, matrix_inverse
import time

def stopping_condition(current_weight:np.ndarray, new_weight) -> bool:
    """
    :param current_weight: weight vector of current iteration
    :param new_weight: updated weight vector using Eq4.92 of [B]
    :return: True if the condition is fulfilled
    """
    num = new_weight - current_weight
    den = current_weight
    return ((num.T @ num) /(den.T @ den))  < 1e-3

def run_newtons_method(feature_matrix: np.ndarray, target_vector: np.ndarray, alpha: float = 0.01, max_iterations: int = 100) -> (
np.ndarray, np.ndarray):
    """
    :param feature_matrix: features matrix of dimension #samples x #features
    :param target_vector: target matrix of dimension #samples x 1
    :param alpha: variance of the prior distribution of weight vector
    :param max_iterations: maximum number of iterations
    :return: MAP weight vector
    """
    N, d = feature_matrix.shape
    iterations = 0
    weight_vector = np.zeros((d, 1))
    hessian_inverse = np.zeros((d, d))

    while iterations < max_iterations:
        prediction_vector = sigmoid(feature_matrix @ weight_vector)
        R = np.identity(N) * (prediction_vector * (1 - prediction_vector))
        hessian_inverse = matrix_inverse(alpha * np.identity(d) + feature_matrix.T @ R @ feature_matrix)
        jacobian = feature_matrix.T @ (prediction_vector - target_vector) + alpha * weight_vector
        new_weight_vector = weight_vector - hessian_inverse @ jacobian

        if iterations > 0:
            if stopping_condition(weight_vector, new_weight_vector):
                weight_vector = new_weight_vector
                prediction_vector = sigmoid(feature_matrix @ weight_vector)
                R = np.identity(N) * (prediction_vector * (1 - prediction_vector))
                hessian_inverse = matrix_inverse(alpha * np.identity(d) + feature_matrix.T @ R @ feature_matrix)
                break
        iterations += 1
        weight_vector = new_weight_vector

    weight_mean, weight_variance = weight_vector, hessian_inverse
    return weight_mean, weight_variance

def add_w0(data_matrix:np.ndarray) -> np.ndarray:
    """
    :param data_matrix: features matrix of size #samples x #features
    :return: adds a column of 1 to the features matrix
    """
    N, d = data_matrix.shape
    w0 = np.ones((N, 1))
    updated_data_matrix = np.hstack((w0, data_matrix))
    return updated_data_matrix

def logistic_regression_training(train_features:np.ndarray, train_labels:np.ndarray, record_time:bool=False):
    """
    :param train_features: features matrix of dimension #samples x #features for training
    :param train_labels: labels vector of dimension #samples x 1 for training
    :param record_time: If True, record the time taken by Newton's Method of finding wmap
    :return: wmap and sn. If record_time=True, return the time as well
    """
    train_features = add_w0(train_features)
    if record_time:
        start_time = time.perf_counter()
        wmap, sn = run_newtons_method(train_features, train_labels)
        newtons_time = time.perf_counter() - start_time
        return wmap, sn, newtons_time
    else:
        return run_newtons_method(train_features, train_labels)

def map_predictions_to_class(x):
    """
    :param x: probability value/vector that the data belongs to class 1
    :return: 1 if p>=0.5, else 0
    """
    return np.floor(x + 0.5)

def logistic_regression_testing_single_datapoint(datapoint:np.ndarray, wmap:np.ndarray, sn:np.ndarray) -> np.ndarray:
    """
    :param datapoint: data vector of dimension #features x 1
    :param wmap: MAP weight vector of dimension #features x 1
    :param sn: covariance matrix of the posterior distribution of weight vector
    :return: prediction for a single datapoint(1 or 0)
    """
    mu_a = wmap.T  @ datapoint
    variance_a = datapoint.T @ sn @ datapoint
    den = 1 + (np.pi * variance_a)/8
    p = sigmoid(mu_a / np.sqrt(den))
    t_hat = map_predictions_to_class(p)
    return t_hat

def logistic_regression_testing(test_features:np.ndarray, wmap:np.ndarray, sn:np.ndarray) -> np.ndarray:
    """
    :param test_features: test data of dimensions #samples x #features
    :param wmap: MAP weight vector of dimension #features x 1
    :param sn: covariance matrix of the posterior distribution of weight vector
    :return: predictions for the test data of dimensions #samples x 1
    """
    t_hat_vector = np.zeros((test_features.shape[0], 1))
    test_features = add_w0(test_features)
    for i, row in enumerate(test_features):
        prediction = logistic_regression_testing_single_datapoint(row.reshape(-1, 1), wmap, sn)
        t_hat_vector[i] = prediction
    return t_hat_vector