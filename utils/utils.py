import numpy as np
import csv
import matplotlib.pyplot as plt
from topic_modelling.config.config import Config

def read_txt_file(path_to_txt_file:str) -> list:
    """
    :param path_to_txt_file: path to text file
    :return: list of words in the text file
    """
    f = open(path_to_txt_file, "r")
    data = f.read()
    words_list = data.split()
    return words_list


def read_classification_data(config:Config) -> dict[str, float]:
    """
    :param dataset_name: name of the dataset
    :return: matrix of dimension #samples x 1
    """
    label_map = {}
    filepath = f"{config.main_data_path}/{config.dataset}/index.csv"
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label_map[row[0]] = float(row[1])
    return label_map

def sigmoid(x):
    """
    :param x: input(single value/vector)
    :return: sigmoid of input
    """
    return 1/(1 + np.exp(-x))

def matrix_inverse(matrix:np.ndarray, regularizer:float=1e-9) -> np.ndarray:
    """
    :param matrix: input matrix
    :param regularizer: do not change this value
    :return: adds a regularizer before inverting so that inverse do not return complex values
    """
    size = matrix.shape[0]
    return np.linalg.inv(matrix + regularizer * np.identity(size))


def generate_and_save_error_plot(dataset_name: str, mean_error: dict[float, dict[str, float]],
                        std_error: dict[float, dict[str, float]]):
    """
    :param dataset_name: name of the dataset on which to compare the 3 models
    :param mean_error: mean error rate for the 3 models on the dataset for different training fractions
    :param std_error: std error rate for the 3 models on the dataset for different training fractions
    :return: plot the learning curve(training fraction vs test error rate) and saves the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=2, sharey=False)
    lda_error, lda_std_error = [], []
    bow_error, bow_std_error = [], []
    lda_time, lda_std_time = [], []
    bow_time, bow_std_time = [], []

    tf = np.arange(0.1, 1.1, 0.1)
    for f in mean_error.keys():
        lda_error.append(mean_error[f]["mean_lda_error"])
        lda_std_error.append(std_error[f]["sd_lda_error"])

        bow_error.append(mean_error[f]["mean_bow_error"])
        bow_std_error.append(std_error[f]["sd_bow_error"])

        lda_time.append(mean_error[f]["mean_lda_time"])
        lda_std_time.append(std_error[f]["sd_lda_time"])

        bow_time.append(mean_error[f]["mean_bow_time"])
        bow_std_time.append(std_error[f]["sd_bow_time"])

    ax[0].plot(tf, lda_error, label="LDA", color='red')
    ax[0].errorbar(tf, lda_error, yerr=lda_std_error, fmt='o', color='red')
    ax[0].plot(tf, bow_error, label="BoW", color='blue')
    ax[0].errorbar(tf, bow_error, yerr=bow_std_error, fmt='o', color='blue')
    ax[0].set_xlabel("Training size fraction")
    ax[0].set_ylabel("Mean test error rate")
    ax[0].legend()
    ax[0].set_title('Error rate LDA vs BoW')
    ax[1].plot(tf, lda_time, label="LDA", color='red')
    ax[1].errorbar(tf, lda_time, yerr=lda_std_time, fmt='o', color='red')
    ax[1].plot(tf, bow_time, label="bow", color='blue')
    ax[1].errorbar(tf, bow_time, yerr=bow_std_time, fmt='o', color='blue')
    ax[1].set_xlabel("Training size fraction")
    ax[1].set_ylabel("Mean runtime(seconds)")
    ax[1].legend()
    ax[1].set_title(f'Runtime LDA vs BoW - dataset {dataset_name}')
    plt.savefig(f"plots/{dataset_name}.png")
