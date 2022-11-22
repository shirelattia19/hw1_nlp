import pickle
from preprocessing import preprocess_train, read_test
from optimization import get_optimal_vector
from inference import tag_all_test
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.metrics import ConfusionMatrixDisplay
import datetime


def calc_accuracy(test_path, predictions_path):
    with open(test_path) as f:
        y = []
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            y = y + line.split(' ')
    with open(predictions_path) as f:
        y_pred = []
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            if line[-3:] == "~_~":
                line = line[:-4]
            y_pred = y_pred + line.split(' ')
    # for a, b in zip(y, y_pred):
    #     if a != b:
    #         print(f'{a}  vs  {b}')
    return list(a == b for a, b in zip(y, y_pred)).count(True) / len(y)


def get_metrics(test_path, feature2id, predictions_path):
    test = read_test(test_path, tagged=True)
    pred = read_test(predictions_path, tagged=True)
    all_tags = sorted(list(feature2id.feature_statistics.tags))
    number_of_tags = len(all_tags)
    dict_tags = {i: all_tags[i] for i in range(number_of_tags)}
    dict_tags_inv = {v: k for k, v in dict_tags.items()}
    confusion_matrix = np.zeros((number_of_tags, number_of_tags))
    true_tags = []
    p_tags = []

    for sen_y, sen_y_pred in zip(test, pred):
        y = sen_y[1]
        gt_tags = y[2:-2]
        true_tags = true_tags + gt_tags
        y_pred = sen_y_pred[1]
        pred_tags = y_pred[2:-2]
        p_tags = p_tags + pred_tags

        for i in range(len(pred_tags)):
            predicted = dict_tags_inv[pred_tags[i]]
            try:
                true = dict_tags_inv[gt_tags[i]]
                confusion_matrix[predicted, true] += 1
            except KeyError:
                pass
    for_worst = confusion_matrix - np.diag(np.diag(confusion_matrix))
    ten_worst = np.argsort(np.sum(for_worst, axis=0))[-10:]
    ten_worst_conf_mat = confusion_matrix[np.ix_(ten_worst, ten_worst)]

    print(ten_worst_conf_mat)

    # cmd_obj = ConfusionMatrixDisplay(ten_worst_conf_mat, display_labels=[dict_tags[i] for i in ten_worst])
    # cmd_obj.plot(cmap='Blues', )
    # cmd_obj.ax_.set(
    #     title='Sklearn Confusion Matrix with labels!!',
    #     xlabel='Predicted Tags',
    #     ylabel='GT Tags')
    #
    # plt.show()


def check_reproducible(test_path, pre_trained_weights, feature2id, predictions_path):
    predictions_path_last = 'predictions_last.wtag'
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path_last)
    accuracy = calc_accuracy(predictions_path_last, predictions_path)
    print(f'Two test1 in a row looks the same at {accuracy * 100} %')


def run_model1():
    # model 1
    threshold = 1
    lam = 0.8
    train_path = f"data/train1.wtag"
    comp_path = f"data/comp1.words"
    test_path = "data/test1.wtag"

    weights_path = f'weights.pkl'
    predictions_path = f'predictions.wtag'
    predictions_comp_path = f'predictions_comp1.wtag'

    time1 = datetime.datetime.now()

    #statistics, feature2id = preprocess_train(train_path, threshold)
    #get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    time2 = datetime.datetime.now()
    print(f'time for features creation: {time2 - time1}')

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    accuracy = calc_accuracy(test_path, predictions_path)
    print(f'Accuracy is {accuracy}\n')
    tag_all_test(comp_path, pre_trained_weights, feature2id, predictions_comp_path)


    get_metrics(test_path, feature2id, predictions_path)
    # check_reproducible(test_path, pre_trained_weights, feature2id, predictions_path)

def run_model2():
    # model 2
    threshold = 1
    lam = 0.01
    train_path = f"data/train2.wtag"
    comp_path = f"data/comp2.words"

    weights_path = f'weights2.pkl'
    predictions_comp_path = f'predictions_comp2.wtag'

    time1 = datetime.datetime.now()

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    time2 = datetime.datetime.now()
    print(f'time for features creation: {time2 - time1}')

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(comp_path, pre_trained_weights, feature2id, predictions_comp_path)

    # check_reproducible(comp_path, pre_trained_weights, feature2id, predictions_path)

def split_data_to_folds(train_path, number_of_folds):
    folds_files = []
    train_files = []
    for fold_num in range(number_of_folds):
        folds_files.append(open(f"test_fold{fold_num}.wtag", "w+"))
        train_files.append(open(f"train_without_fold{fold_num}.wtag", "w+"))
    with open(train_path) as f:
        for index, line in enumerate(f):
            current_fold = index % number_of_folds
            fold_file = folds_files[current_fold]
            fold_file.write(line)
            for fold_num in range(number_of_folds):
                if fold_num != current_fold:
                    train_file = train_files[fold_num]
                    train_file.write(line)
    for fold, train in zip(folds_files, train_files):
        fold.close()
        train.close()


def run_model2_cross_validation():
    # model 2
    thresholds = [1, 2, 3, 4]
    lams = [0.01, 0.5, 0.7, 0.8, 1]
    Bs = [2, 3, 4]
    number_of_folds = 5
    train_path = f"data/train2.wtag"

    weights_path = f'weights2.pkl'
    predictions_path = f'predictions2.wtag'
    my_text = ''
    accuracies = {}
    split_data_to_folds(train_path, number_of_folds)
    for fold_num in range(number_of_folds):
        for threshold in thresholds:
            fold_train_path = f"train_without_fold{fold_num}.wtag"
            fold_test_path = f"test_fold{fold_num}.wtag"
            statistics, feature2id = preprocess_train(fold_train_path, threshold)

            for lam in lams:
                get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
                with open(weights_path, 'rb') as f:
                    optimal_params, feature2id = pickle.load(f)
                pre_trained_weights = optimal_params[0]
                print(pre_trained_weights)

                for B in Bs:
                    tag_all_test(fold_test_path, pre_trained_weights, feature2id, predictions_path)
                    accuracy = calc_accuracy(fold_test_path, predictions_path)
                    print(f'Accuracy is {accuracy}\n')
                    if (threshold, lam, B) in accuracies:
                        accuracies[(threshold, lam, B)] += accuracy
                    else:
                        accuracies[(threshold, lam, B)] = accuracy

    for threshold, lam, B in accuracies.keys():
        my_text += f"For threshold={threshold}, lam={lam}, B={B}\n" \
                   f"Mean accuracy on {number_of_folds} folds is={accuracies[(threshold, lam, B)] / number_of_folds}\n\n"
    print(my_text)


def main():
    run_model1()
    run_model2_cross_validation()
    run_model2()


if __name__ == '__main__':
    main()
