from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple

WORD = 0
TAG = 1


# TODO: implement some other features

class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "digits", "capitals",
                             "tiret", "double", "point", 'ed', 'ing']  # the feature classes used in the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the text

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                    # f100
                    if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1

                    # f101
                    suffixes = set(cur_word[-1 * i:] for i in range(1, 5))
                    for suf in suffixes:
                        if (suf, cur_tag) not in self.feature_rep_dict["f101"]:
                            self.feature_rep_dict["f101"][(suf, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101"][(suf, cur_tag)] += 1

                    # f102
                    prefixes = set(cur_word[:i] for i in range(1, 5))
                    for pre in prefixes:
                        if (pre, cur_tag) not in self.feature_rep_dict["f102"]:
                            self.feature_rep_dict["f102"][(pre, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102"][(pre, cur_tag)] += 1

                    t_1 = '*' if word_idx == 0 else split_words[word_idx - 1].split('_')[1]
                    t_2 = '*' if word_idx <= 1 else split_words[word_idx - 2].split('_')[1]

                    # f103
                    if (t_2, t_1, cur_tag) not in self.feature_rep_dict["f103"]:
                        self.feature_rep_dict["f103"][(t_2, t_1, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f103"][(t_2, t_1, cur_tag)] += 1

                    # f104
                    if (t_1, cur_tag) not in self.feature_rep_dict["f104"]:
                        self.feature_rep_dict["f104"][(t_1, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f104"][(t_1, cur_tag)] += 1

                    # f105
                    if cur_tag not in self.feature_rep_dict["f105"]:
                        self.feature_rep_dict["f105"][cur_tag] = 1
                    else:
                        self.feature_rep_dict["f105"][cur_tag] += 1

                    pre_word = '*' if word_idx == 0 else split_words[word_idx - 1].split('_')[0]

                    # f106
                    if (pre_word, cur_tag) not in self.feature_rep_dict["f106"]:
                        self.feature_rep_dict["f106"][(pre_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f106"][(pre_word, cur_tag)] += 1

                    next_word = "~" if word_idx == len(split_words) - 1 else split_words[word_idx + 1].split('_')[0]

                    # f107
                    if (next_word, cur_tag) not in self.feature_rep_dict["f107"]:
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] += 1

                    # feature for capital letters
                    if cur_word[0].isupper():
                        if cur_tag not in self.feature_rep_dict["capitals"]:
                            self.feature_rep_dict["capitals"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["capitals"][cur_tag] += 1

                    # feature for digits
                    if any(c.isdigit() for c in cur_word):
                        if cur_tag not in self.feature_rep_dict["digits"]:
                            self.feature_rep_dict["digits"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["digits"][cur_tag] += 1

                    # '-' in the word
                    if len(cur_word) > 2 and '-' in cur_word[1:-1]:
                        if cur_tag not in self.feature_rep_dict["tiret"]:
                            self.feature_rep_dict["tiret"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["tiret"][cur_tag] += 1

                    # '.' in the word but not alone
                    if len(cur_word) > 1 and '.' in cur_word:
                        if cur_tag not in self.feature_rep_dict["point"]:
                            self.feature_rep_dict["point"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["point"][cur_tag] += 1

                    # double letters
                    if len(cur_word) > 1 and True in [cur_word[i] == cur_word[i + 1] for i in range(len(cur_word) - 1)]:
                        if t_1 not in self.feature_rep_dict["double"]:
                            self.feature_rep_dict["double"][t_1] = 1
                        else:
                            self.feature_rep_dict["double"][t_1] += 1

                    # 'ed' words
                    if cur_word.endswith('ed'):
                        if t_1 not in self.feature_rep_dict["ed"]:
                            self.feature_rep_dict["ed"][t_1] = 1
                        else:
                            self.feature_rep_dict["ed"][t_1] += 1

                    # 'ing' words
                    if cur_word.endswith('ing'):
                        if t_1 not in self.feature_rep_dict["ing"]:
                            self.feature_rep_dict["ing"][t_1] = 1
                        else:
                            self.feature_rep_dict["ing"][t_1] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {fd: OrderedDict() for fd in self.feature_statistics.feature_rep_dict.keys()}
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time (threshold times) in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) \
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word = history[0]
    c_tag = history[1]
    features = []

    # f100
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    # f101
    suffixes = set(c_word[-1 * i:] for i in range(1, 5))
    for suf in suffixes:
        if (suf, c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(suf, c_tag)])

    # f102
    prefixes = set(c_word[:i] for i in range(1, 5))
    for pre in prefixes:
        if (pre, c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(pre, c_tag)])

    p_tag = history[3]
    pp_tag = history[5]

    # f103
    if (pp_tag, p_tag, c_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(pp_tag, p_tag, c_tag)])

    # f104
    if (p_tag, c_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(p_tag, c_tag)])

    # f105
    if c_tag in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][c_tag])

    p_word = history[2]
    n_word = history[6]

    # f106
    if (p_word, c_tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(p_word, c_tag)])

    # f107
    if (n_word, c_tag) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(n_word, c_tag)])

    # feature for capital letters
    if c_word[0].isupper() and c_tag in dict_of_dicts["capitals"]:
        features.append(dict_of_dicts["capitals"][c_tag])

    # feature for digits
    if any(c.isdigit() for c in c_word) and c_tag in dict_of_dicts["digits"]:
        features.append(dict_of_dicts["digits"][c_tag])

    # feature for '-'
    if len(c_word) > 2 and '-' in c_word[1:-1] and c_tag in dict_of_dicts["tiret"]:
        features.append(dict_of_dicts["tiret"][c_tag])

    # feature for '.'
    if len(c_word) > 1 and '.' in c_word and c_tag in dict_of_dicts["point"]:
        features.append(dict_of_dicts["point"][c_tag])

    # double letters in a word
    if len(c_word) > 1 and True in [c_word[i] == c_word[i + 1] for i in range(len(c_word) - 1)] \
            and p_tag in dict_of_dicts["double"]:
        features.append(dict_of_dicts["double"][p_tag])

    # ed words
    if c_word.endswith('ed') and p_tag in dict_of_dicts["ed"]:
        features.append(dict_of_dicts["ed"][p_tag])

    # ing words
    if c_word.endswith('ing') and p_tag in dict_of_dicts["ing"]:
        features.append(dict_of_dicts["ing"][p_tag])

    return features


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
