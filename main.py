import pickle
from preprocessing import preprocess_train, read_test
from optimization import get_optimal_vector
from inference import tag_all_test
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as c_f
from sklearn.metrics import ConfusionMatrixDisplay


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
    # Selecting ten worst tags
    ten_worst = np.argsort(np.sum(for_worst, axis=0))[-10:]
    ten_worst_conf_mat = confusion_matrix[np.ix_(ten_worst, ten_worst)]
    # Computing Accuracy
    accuracy = 100 * np.trace(confusion_matrix) / np.sum(confusion_matrix)
    # print("Model " + str(num_model) + " Accuracy.: " + str(accuracy) + " %")
    print("Accuracy.: " + str(accuracy) + " %")
    print("Ten Worst Elements: " + str([dict_tags[i] for i in ten_worst]))
    print("Confusion Matrix:")
    print(ten_worst_conf_mat)

    cmd_obj = ConfusionMatrixDisplay(ten_worst_conf_mat, display_labels=[dict_tags[i] for i in ten_worst])
    cmd_obj.plot(cmap='Blues', )
    cmd_obj.ax_.set(
        title='Sklearn Confusion Matrix with labels!!',
        xlabel='Predicted Tags',
        ylabel='GT Tags')

    plt.show()

def main():
    threshold = 1
    lam = 1

    train_path = "data/train2.wtag"
    # test_path = "data/comp1.words"
    test_path = "data/test1.wtag"

    weights_path = 'weights2.pkl'
    predictions_path = 'predictions.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    accuracy = calc_accuracy(test_path,  predictions_path)
    print(f'Accuracy is {accuracy}')
    get_metrics(test_path, feature2id, predictions_path)

    # for B in range(2, 6):
    #     print(f'-------------------------------------- Running test1 for B={B} --------------------------------------')
    #     #tag_all_test(test_path, pre_trained_weights, feature2id, f'predictionsB{B}.wtag', B)
    #     accuracy = calc_accuracy(test_path,  f'predictionsB{B}.wtag')
    #     print(f'Accuracy fo B={B} is {accuracy}')


if __name__ == '__main__':
    main()
