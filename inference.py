from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import numpy as np


def memm_viterbi(sentence, pre_trained_weights, feature2id, B=2):
    # TODO : implement this function
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
            for v in Sk:
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
                    #sum_v_fs = np.sum(np.exp(list(v_fs.values())))

                    q = (v_fs[v] - log_v_fs)
                    #q = (v_fs[v] / sum_v_fs)
                    if (t,u) in pi[k-2].keys():
                        tag_probabilities[(t, u, v)] = pi[k - 2][(t, u)] + q
                        #tag_probabilities[(t, u, v)] = pi[k - 2][(t, u)] * q



        sorted_prob = sorted(tag_probabilities.items(), key=lambda g: g[1], reverse=True)
        Sk = []
        for count, ((t, u, v), prob) in enumerate(sorted_prob):
            if count == B:
                break
            Sk.append(v)
            if len(pi) > k-1:
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
    t.append('~')
    return t[1:]

    # TODO add pi for ~


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
