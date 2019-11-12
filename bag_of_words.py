import sys
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import getopt
import csv
nltk.download('punkt')
nltk.download('stopwords')


class VocabularyFrequency:
    """
    Represents the frequency of the words in the comments.
    The word_frequency attribute is an array in which the rows are words and the columns are categories.
    The last 2 columns of this array are the general frequency of the word accessing through [-2],
    and the index of the word in the vocabulary.
    """
    def __init__(self, word_frequency, vocabulary):
        self.word_frequency = word_frequency
        self.vocabulary = vocabulary


class CategoryStats:
    """
    Represent the statistics of the categories
    """
    def __init__(self, categories, category_priors):
        self.categories = categories
        self.category_priors = category_priors


def get_key_by_value(dictionary, value):
    """
    Searches a key by value in a dictionary
    :param dictionary: key-value structure. In this case the value must be unique.
    :param value: the value to be searched
    :return: the key that contains the value.
    """
    for (k, v) in dictionary.items():
        if value == v:
            return k


def remove_most_and_less_frequent(data_set, top_indexes, bottom_indexes):
    result = np.delete(data_set, top_indexes, axis=0)
    return np.delete(result, bottom_indexes, axis=0)


def reindex_vocabulary(vocabulary, data_set):
    """
    Reindex the vocabulary
    :param vocabulary:
    :param data_set:
    :return:
    """
    temp_dict = {current_index: new_index for new_index, current_index in enumerate(data_set[:, -1])}
    return {k: temp_dict[v] for (k, v) in vocabulary.items() if v in temp_dict}


def load_train_data(file_path):
    """
    :param file_path:
    :return: an array[n,2] with comments and their topic
    """
    train = np.load(file_path, allow_pickle=True)
    return np.array([train[0], train[1]]).T


def load_test(file_path):
    """
    :param file_path:
    :return: an list of comments
    """
    return np.load(file_path, allow_pickle=True)


def generate_frequency(data_set, validation_size):
    stop_words = set(stopwords.words('english'))
    cat_count = len(np.unique(data_set[:, 1]))
    frequencies = []
    dict_words = {}
    categories = dict([(cat, index) for index, cat in enumerate(np.unique(data_set[:, 1]))])
    for i in range(len(data_set[:len(data_set) - validation_size])):
        comment_words = nltk.word_tokenize(re.sub(r'[^A-Za-z0-9]', ' ', data_set[i, 0].lower()))
        for word in comment_words:
            if word not in stop_words:
                if word not in dict_words:
                    dict_words[word] = len(dict_words)
                    frequencies.append([0.] * (cat_count + 2))
                    frequencies[dict_words[word]][-1] = dict_words[word]

                frequencies[dict_words[word]][-2] += 1
                frequencies[dict_words[word]][categories[data_set[i, 1]]] += 1

    return VocabularyFrequency(np.array(frequencies), dict_words)


def get_category_stats(data_set):
    categories = dict([(cat, index) for index, cat in enumerate(np.unique(data_set[:, 1]))])

    def calculate_prior():
        # cat_stat = probabilite d'une categorie dans tous les documents
        cat_stat = []
        for key, value in categories.items():
            cat_stat.append([value, np.sum(data_set[:, 1] == key) / data_set.shape[0]])
        return np.array(cat_stat)
    return CategoryStats(categories, calculate_prior())


def get_data(file_path):
    train = np.load(file_path, allow_pickle=True)


def apply_laplace_smoothing(data_set, alpha):
    """
    Apply Laplace Smoothing
    :param data_set:
    :param alpha:
    :return:
    """
    smoothed_freq = data_set[:, :-2] + alpha
    # stat = probabilité d'un mot étant donné une categorie
    return smoothed_freq / np.sum(smoothed_freq, axis=0)


def predict(train_set, test_set, vocabulary_frequency, category_statistics):
    result_test = []
    cat_stat = category_statistics.category_priors
    categories = category_statistics.categories
    voc = vocabulary_frequency.vocabulary
    voc_stat = vocabulary_frequency.word_frequency
    for comment in test_set:
        cleaned = nltk.word_tokenize(re.sub(r'[^A-Za-z0-9]', ' ', comment.lower()))
        prediction = np.array([[1.] * len(cleaned)] * len(cat_stat))
        for i, (cat, index) in enumerate(categories.items()):
            for j, w in enumerate(cleaned):
                if w in voc:
                    prediction[i, j] = voc_stat[voc[w], index] if voc_stat[voc[w], index] > 0.0 else 1 / len(voc)
                else:
                    prediction[i, j] = 1. / len(voc)
        cat_prediction = np.argmax(np.sum(np.log(prediction), axis=1) + np.log(cat_stat[:, 1]))
        result_test.append(get_key_by_value(categories, cat_prediction))
    return result_test


def remove_words(vocabulary_frequency):
    """
        Remove words and reindex the vocabulary
        :param vocabulary_frequency:
        :return: tuple of data_set and vocabulary
    """
    indexes = np.argsort(vocabulary_frequency.word_frequency[:, -2])
    top_indexes = indexes[:300]
    bottom_indexes = indexes[indexes.shape[0] - 45:]

    freq_norm = remove_most_and_less_frequent(vocabulary_frequency.word_frequency, top_indexes, bottom_indexes)
    vocabulary = reindex_vocabulary(vocabulary_frequency.vocabulary, freq_norm)
    return VocabularyFrequency(freq_norm, vocabulary)


def train_model(train_file_path, alpha, validation_size):
    train_set = load_train_data(train_file_path)
    category_stats = get_category_stats(train_set)
    vocabulary_frequency = remove_words(generate_frequency(train_set, validation_size))
    vocabulary_frequency_ = VocabularyFrequency(apply_laplace_smoothing(vocabulary_frequency.word_frequency, alpha),
                                                vocabulary_frequency.vocabulary)
    return train_set, vocabulary_frequency_, category_stats


def generate_cvs(output_file, prediction):
    with open(output_file, mode='w') as csv_file:
        fieldnames = ['Id', 'Category']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, result in enumerate(prediction):
            writer.writerow({'Id': i, 'Category': result})


def evaluate_validation(train_set, validation_size, vocabulary_frequency, category_stats):
    validation_set = train_set[train_set.shape[0] - validation_size:, 0]
    labels = train_set[train_set.shape[0] - validation_size:, 1]
    prediction = predict(train_set, validation_set, vocabulary_frequency, category_stats)
    return np.sum(prediction == labels), np.sum(prediction == labels) / len(validation_set)


def get_opt(opts):
    params = {}
    for i in opts:
        params[i[0]] = i[1]
    return params


def main(argv):
    opts, args = getopt.getopt(argv, "hi:t:a:o:v", ["itrain_file=", "ttest_file=", "aalpha=", "ooutput_csv_file=", "vvalidation_size="])
    params = get_opt(opts)
    validation_size = params.get('-v', 0)
    train_set, vocabulary_frequency, category_stats = train_model(params['-i'], float(params['-a']), validation_size)
    if validation_size == 0:
        test_set = load_test(params['-t'])
        prediction = predict(train_set, test_set, vocabulary_frequency, category_stats)
        generate_cvs(params['-o'], prediction)
    else:
        total, rate = evaluate_validation(train_set, validation_size, vocabulary_frequency, category_stats)
        print('Accuracy: ', total, rate)


if __name__ == "__main__":
    main(sys.argv[1:])
