import pickle
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple

WORD = 0
TAG = 1


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

def memm_viterbi(sentence, pre_trained_weights, feature2id, B=2):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    pi = [{('*', '*'): 1}]
    bp = [{('*', '*'): '*'}]
    Sk_2 = {'*'}
    Sk_1 = {'*'}
    n = len(sentence)
    for k in range(2, n - 1):
        Sk = set(feature2id.feature_statistics.tags)
        tag_probabilities = {}
        for u in Sk_1:
            for t in Sk_2:
                v_fs = {}
                for y_prime_tag in Sk:
                    features_indexes = set()
                    v_f = 0
                    hist = (sentence[k], y_prime_tag, sentence[k - 1], u, sentence[k - 2], t, sentence[k + 1])
                    features_indexes = features_indexes.union(
                        set(represent_input_with_features(hist, feature2id.feature_to_idx)))
                    for index in features_indexes:
                        v_f += pre_trained_weights[index]
                    v_fs[y_prime_tag] = v_f
                log_v_fs = np.log(np.sum(np.exp(list(v_fs.values()))))
                # sum_v_fs = np.sum(np.exp(list(v_fs.values())))
                for v in Sk:
                    q = (v_fs[v] - log_v_fs)
                    # q = (v_fs[v] / sum_v_fs)
                    if (t, u) in pi[k - 2].keys():
                        tag_probabilities[(t, u, v)] = pi[k - 2][(t, u)] + q
                        # tag_probabilities[(t, u, v)] = pi[k - 2][(t, u)] * q

        sorted_prob = sorted(tag_probabilities.items(), key=lambda g: g[1], reverse=True)
        Sk = []
        for count, ((t, u, v), prob) in enumerate(sorted_prob):
            if len(pi) > k - 1 and len(pi[k - 1]) == B:
                break
            Sk.append(v)
            if len(pi) > k - 1:
                if (u, v) not in pi[k - 1].keys():
                    pi[k - 1][(u, v)] = prob
                    bp[k - 1][(u, v)] = t
            else:
                pi.append({(u, v): prob})
                bp.append({(u, v): t})
        Sk_2 = Sk_1
        Sk_1 = set(Sk)
    tn_1, tn = max(pi[-1], key=pi[-1].get)
    t = []
    t.append(tn)
    t.append(tn_1)
    for k in range(2, n - 1):
        t.append(bp[n - k - 1][(t[k - 1], t[k - 2])])
    t = list(reversed(t))
    return t[1:]


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, B=2):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "w+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, B)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()

def main():
    comp_path1 = f"data/comp1.words"
    weights_path1 = f'weights.pkl'
    predictions_comp_path1 = f'comp_m1_342517406_209948728.wtag'
    with open(weights_path1, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    tag_all_test(comp_path1, pre_trained_weights, feature2id, predictions_comp_path1)

    comp_path2 = f"data/comp2.words"
    weights_path2 = f'weights2.pkl'
    predictions_comp_path2 = f'comp_m2_342517406_209948728.wtag'
    with open(weights_path2, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    tag_all_test(comp_path2, pre_trained_weights, feature2id, predictions_comp_path2)

if __name__ == '__main__':
    main()