import numpy as np

from topic_modelling.config.config import Config
from topic_modelling.utils.logistic_regression import logistic_regression_training, logistic_regression_testing
from topic_modelling.LDA import prepare_lda_bow_features_and_labels_for_classification

def evallassification_task(prediction:np.ndarray, truth:np.ndarray) -> float:
    """
    :param prediction: prediction vector of dimension #samples x 1
    :param truth: actual value vector of dimension #samples x 1
    :return: error rate
    """
    error = np.abs(prediction - truth)
    return np.mean(error)


def split_test_train_data(lda_features: np.ndarray, bow_features: np.ndarray, labels: np.ndarray,
                          fraction: float = 1 / 3) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    :param lda_features: lda representation of documents
    :param bow_features: bow representation of documents
    :param labels: label matrix of dimensions #samples x 1
    :param fraction: fraction of data to be used as testing data
    :return: features for training, labels for training, features for testing, labels for testing for both lda and bow
    """
    N = labels.shape[0]
    test_size = int(N * fraction)
    test_indices = np.random.choice(N, size=test_size, replace=False)
    train_indices = np.setdiff1d(np.arange(N), test_indices)

    np.random.shuffle(train_indices)
    test_lda_data = lda_features[test_indices]
    test_bow_data = bow_features[test_indices]
    test_labels = labels[test_indices]

    train_lda_data = lda_features[train_indices]
    train_bow_data = bow_features[train_indices]
    train_labels = labels[train_indices]
    return train_lda_data, train_bow_data, train_labels, test_lda_data, test_bow_data, test_labels

def train_and_evaluate_logistic_regression_task(train_features:np.ndarray, train_labels:np.ndarray, test_features:np.ndarray, test_labels:np.ndarray) -> (float, float):
    """
    :param train_features: feature matrix for training of dimensions #training_samples x #features
    :param train_labels: label vector for training of dimensions #training_samples x 1
    :param test_features: feature matrix for testing of dimensions #testing_samples x #features
    :param test_labels: label vector for testing of dimensions #testing_samples x 1
    :return: error rate of the LR model on the test set
    """
    w, s, run_time = logistic_regression_training(train_features, train_labels, record_time=True)
    lr_prediction = logistic_regression_testing(test_features, w, s)
    lr_error = evaluate_classification_task(lr_prediction, test_labels)
    return lr_error, run_time


def model_training_and_evaluation_for_one_training_fraction(runs:int, fraction: float, lda_data: np.ndarray, bow_data,
                                                                target: np.ndarray) -> dict[str, list]:
    """
    :param fraction: Fraction of training data required. Should be in [0, 1]
    :param lda_data: lda feature representation of documents
    :param bow_data: bow feature representation of documents
    :param target: label vector of dimensions #samples x 1
    :return: error rate of LR and GM(shared and non-shared generative model) on the test set
    """
    result = {"lda_error": [], "bow_error": [], "lda_time": [], "bow_time": []}

    for run in range(runs):
        train_lda_features, train_bow_features, train_labels, test_lda_features, test_bow_features, test_labels = split_test_train_data(
            lda_data, bow_data, target)

        N = train_labels.shape[0]
        train_size_run = int(fraction * N)
        train_lda_features_for_run = train_lda_features[:train_size_run, :]
        train_bow_features_for_run = train_bow_features[:train_size_run, :]
        train_labels_for_run = train_labels[:train_size_run, :]

        lda_error, lda_time = train_and_evaluate_logistic_regression_task(train_lda_features_for_run,
                                                                          train_labels_for_run, test_lda_features,
                                                                          test_labels)

        bow_error, bow_time = train_and_evaluate_logistic_regression_task(train_bow_features_for_run,
                                                                          train_labels_for_run, test_bow_features,
                                                                          test_labels)

        result["lda_error"].append(lda_error)
        result["bow_error"].append(bow_error)
        result["lda_time"].append(lda_time)
        result["bow_time"].append(bow_time)

    return result

def calculate_mean_error(error_data:dict[str, list])-> dict[str, float]:
    """
    :param error_data:dictionary containing error rates of the models for a particular training fraction for 30 runs
    :return: dictionary containing the mean error rate of 30 runs for the 3 models
    """
    mean_error_dict = {}
    for key in error_data.keys():
        mean_error_dict[f"mean_{key}"] = np.mean(error_data[key])
    return mean_error_dict

def calculate_std_error(error_data:dict[str, list])-> dict[str, float]:
    """
    :param error_data:dictionary containing error rates of the models for a particular training fraction for 30 runs
    :return: dictionary containing the standar deviation of error rates of 30 runs for the 3 models
    """
    sd_error_dict = {}
    for key in error_data.keys():
        sd_error_dict[f"sd_{key}"] = np.std(error_data[key])
    return sd_error_dict


def model_comparison_for_a_dataset(doc_lda_rep, doc_bow_rep, config:Config, starting_tf: float = 0.1,
                                       stopping_tf: float = 1.0, step: float = 0.1) -> (
dict[str, float], dict[float, dict[str, float]]):
    """
    :param doc_lda_rep: lda feature representation of documents
    :param doc_bow_rep: bow feature representation of documents
    :param config: configuration parameters
    :param starting_tf: starting fraction of training data required. Should be in [0, 1]
    :param stopping_tf: stopping fraction of training data required. Should be in [0, 1]. starting_tf <= stopping_tf
    :param step: steps of fraction. Should be in [0, 1]
    :return: mean and standard deviation of error rates of the models on different training fractions
    """
    lda_features, bow_features, target = prepare_lda_bow_features_and_labels_for_classification(doc_lda_rep,
                                                                                                doc_bow_rep, config)
    mean_error_dict = {}
    std_error_dict = {}

    for training_size_fraction in np.arange(start=starting_tf, stop=stopping_tf + step, step=step):
        error_data = model_training_and_evaluation_for_one_training_fraction(config.runs, round(training_size_fraction, 2),
                                                                                 lda_features, bow_features, target)
        mean_error_dict[f"{round(training_size_fraction, 2)}"] = calculate_mean_error(error_data)
        std_error_dict[f"{round(training_size_fraction, 2)}"] = calculate_std_error(error_data)

    return mean_error_dict, std_error_dict