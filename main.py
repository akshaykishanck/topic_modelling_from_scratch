from LDA import *
from utils.evaluation  import model_comparison_for_a_dataset

if __name__ == "__main__":
    config = Config()
    number_of_topics, alpha, beta = config.get_hyper_parameters()
    wi_map, di_map, topic_vector, words_vector, document_vector = gibbs_sampling(config)

    iw_dict = {value: key for key, value in wi_map.items()}

    save_top_words_and_topics(words_vector, iw_dict)

    doc_lda_feature = {}
    for doc in di_map.keys():
        doc_lda_feature[doc] = create_LDA_features_for_a_document(di_map[doc], document_vector, number_of_topics, alpha)

    doc_bow_feature = {}
    for doc in di_map.keys():
        doc_bow_feature[doc] = create_bow_features_for_a_document(doc, wi_map, config.dataset)

    mean_error, standard_error = model_comparison_for_a_dataset(doc_lda_feature, doc_bow_feature, config)
    generate_and_save_error_plot(config.dataset, mean_error, standard_error)

