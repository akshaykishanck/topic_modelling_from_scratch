import os
from topic_modelling.utils.utils import *
from topic_modelling.config.config import Config

def create_word_and_document_index_maps(dataset_path:str, dataset_name: str) -> (dict[str, int], dict[str, int], list, list):
    """
    :param dataset_path: path to the data folder
    :param dataset_name: name of the dataset
    :return: word index map, document-index map, w_n and d_n vectors
    """
    w_n = []
    d_n = []
    filepaths = [entry.path for entry in os.scandir(os.path.join(os.getcwd(), dataset_path, dataset_name)) if
                 "index" not in entry.path]
    doc_index_map = {}
    for i, filepath in enumerate(filepaths):
        doc_number = filepath.split("/")[-1]
        doc_index_map[doc_number] = i
        doc = read_txt_file(filepath)
        w_n += doc
        d_n += [i] * len(doc)

    word_index_map = {}

    for i, word in enumerate(list(set(w_n))):
        word_index_map[word] = i

    updated_w_n = [word_index_map[i] for i in w_n]
    return word_index_map, doc_index_map, updated_w_n, d_n


def initialize_dt_tw_matrix(number_of_topics, topic_vector, words_vector, document_vector):
    """
    :param number_of_topics: number of topics
    :param topic_vector: topic vector
    :param words_vector: words vector
    :param document_vector: document vector
    :return: topic-word matrix and document-topic matrix
    """
    tw_matrix = np.zeros((number_of_topics, len(set(words_vector))))
    td_matrix = np.zeros((number_of_topics, len(set(document_vector))))

    for i, w in enumerate(words_vector):
        tw_matrix[topic_vector[i]][w] += 1

    for i, d in enumerate(document_vector):
        td_matrix[topic_vector[i]][d] += 1

    return tw_matrix, td_matrix.T

def gibbs_sampling(config:Config):
    wi_map, di_map, corpus, docs = create_word_and_document_index_maps(config.main_data_path, config.dataset)

    V = len(set(corpus))
    N_words = len(corpus)
    topics_vector = [np.random.randint(0, config.number_of_topics) for _ in range(N_words)]
    p = np.zeros(config.number_of_topics)
    tw_matrix, dt_matrix = initialize_dt_tw_matrix(config.number_of_topics, topics_vector, corpus, docs)

    pi_n = np.arange(N_words)
    np.random.shuffle(pi_n)

    sum_tw = tw_matrix.sum(axis=1)
    sum_dt = dt_matrix.sum(axis=1)

    for i in range(config.number_of_iterations):

        for n in range(N_words):
            topic = topics_vector[pi_n[n]]
            word = corpus[pi_n[n]]
            doc = docs[pi_n[n]]
            dt_matrix[doc][topic] -= 1
            tw_matrix[topic][word] -= 1

            sum_tw[topic] -= 1
            sum_dt[doc] -= 1

            for k in range(config.number_of_topics):
                n_k = (tw_matrix[k][word] + config.beta) * (dt_matrix[doc][k] + config.alpha)
                d_k = (V * config.beta + sum_tw[k]) * (config.number_of_topics * config.alpha + sum_dt[doc])
                p[k] = n_k / d_k

            p = p / np.sum(p)
            topic = np.random.choice(config.number_of_topics, p=p)
            topics_vector[pi_n[n]] = topic
            dt_matrix[doc][topic] += 1
            tw_matrix[topic][word] += 1

            sum_tw[topic] += 1
            sum_dt[doc] += 1

    return wi_map, di_map, topics_vector, tw_matrix, dt_matrix


def prepare_lda_bow_features_and_labels_for_classification(doc_lda_feature, doc_bow_feature, config:Config):
    """
    :param doc_lda_feature: LDA feature representation of documents
    :param doc_bow_feature: BOW feature representation of documents
    :param config: configuration parameters
    :return:
    """

    doc_label_map = read_classification_data(config)
    V = len(list(doc_bow_feature.values())[0])

    lda_features = np.zeros((len(doc_label_map.keys()), config.number_of_topics))
    bow_features = np.zeros((len(doc_label_map.keys()), V))

    labels = np.zeros((len(doc_label_map.keys()), 1))

    for i, (key, value) in enumerate(doc_label_map.items()):
        lda_features[i] = doc_lda_feature[key]
        bow_features[i] = doc_bow_feature[key]
        labels[i] = value

    return lda_features, bow_features, labels

def calculate_tw_probability(tw_matrix):
    """
    :param tw_matrix: topic-word count matrix
    :return: topic-word probability matrix
    """
    tw_prob = np.zeros(tw_matrix.shape)
    sum_t = tw_matrix.sum(axis=1)
    for i, _ in enumerate(tw_matrix):
        tw_prob[i] = tw_matrix[i]/sum_t[i]
    return tw_prob

def create_LDA_features_for_a_document(doc, dt_matrix, K, alpha):
    feature = np.zeros(K)
    dt_sum = dt_matrix.sum(axis=1)
    for k in range(K):
        feature[k] = (dt_matrix[doc][k] + alpha)/(K*alpha + dt_sum[doc])
    return feature

def create_bow_features_for_a_document(doc, wimap, dataset, data_folder_path="data/"):
    words_list = read_txt_file(os.path.join(os.getcwd(), data_folder_path, dataset, doc))
    bow_feature = np.zeros(len(wimap.keys()))
    for word in words_list:
        bow_feature[wimap[word]] += 1
    return bow_feature

def save_top_words_and_topics(tw_matrix, iw_map):
    """
    :param tw_matrix: topic-word count matrix
    :param iw_map: index-word map
    :return: None. Save top 5 words per topic to csv
    """
    tw_prob_matrix = calculate_tw_probability(tw_matrix)
    ti = np.argsort(-tw_prob_matrix, axis=1)[:, :5]

    with open('topicwords.csv', 'w', newline='') as csvfile:
        fieldnames = ['topic', 'word', 'probability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(ti):
            for x in row:
                writer.writerow({"topic": i, "word": iw_map[x], "probability": tw_prob_matrix[i][x]})





