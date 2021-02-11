from matplotlib import pyplot as plt
from nltk import ngrams
import os

spam_mark = "spmsg"
legit_mark = "legit"
mark_class = {spam_mark: 0, legit_mark: 1}


def read_files(n):
    messages = []
    for num in range(1, 11):
        data_dir = "messages/part" + str(num)
        cur_messages = []
        for filename in os.listdir(data_dir):
            file = open(data_dir + "/" + filename)
            title_words = file.readline()[9:].split(' ')
            title_words = make_n_grams(n, title_words) if n > 1 else title_words
            file.readline()
            message_words = file.readline().split(' ')
            message_words = make_n_grams(n, message_words) if n > 1 else message_words
            if filename.find(spam_mark) > -1:
                cur_messages.append([spam_mark, title_words + message_words])
            else:
                cur_messages.append([legit_mark, title_words + message_words])
        messages.append(cur_messages)
    return messages


def make_n_grams(n, array):
    n_grams = ngrams(array, n)
    arr_res = []
    for el in n_grams:
        res = ""
        for j in el:
            res += str(j) + " "
        arr_res.append(res[:-1])
    return arr_res


def bayes(k, messages_train, messages_test, alpha, lambdas, roc):
    class_count = [0 for _ in range(k)]
    words = {}
    n_train = len(messages_train)
    for i in range(n_train):
        tmp = messages_train[i]
        class_count[mark_class[tmp[0]]] += 1
        tmp_Dict = {}
        for word in tmp[1]:
            if word.strip() not in words:
                words[word.strip()] = [0 for _ in range(k)]
            if word.strip() not in tmp_Dict:
                words[word.strip()][mark_class[tmp[0]]] += 1
            tmp_Dict[word.strip()] = 1

    matrix = {}
    for word in words:
        tmp_word_classes = []
        for cur_cl in range(k):
            tmp_word_classes.append((words[word][cur_cl] + alpha) / (class_count[cur_cl] + 2 * alpha))
        matrix[word] = tmp_word_classes

    n_test = len(messages_test)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(n_test):
        tmp = messages_test[i]
        tmp_Dict = {}
        for word in tmp[1]:
            tmp_Dict[word.strip()] = 1

        sum_p_x_cl = 0
        all_p_x_cl = []
        for cur_cl in range(k):
            p_x_cl = class_count[cur_cl] / n_train
            for word in words:
                p_x_cl *= matrix[word][cur_cl] if (word in tmp_Dict) else (1 - matrix[word][cur_cl])
            p_x_cl *= lambdas[cur_cl]
            sum_p_x_cl += p_x_cl
            all_p_x_cl.append(p_x_cl)

        if sum_p_x_cl != 0:
            probability_spam = round((all_p_x_cl[0]) / sum_p_x_cl, 10)
            probability_legit = round((all_p_x_cl[1]) / sum_p_x_cl, 10)
            cur_predict = spam_mark if probability_spam > probability_legit else legit_mark
            roc.append([probability_legit, tmp[0]])
            if cur_predict == tmp[0]:
                if cur_predict == legit_mark:
                    TP += 1
                else:
                    TN += 1
            else:
                if cur_predict == legit_mark:
                    FP += 1
                else:
                    FN += 1

    return TP, TN, FP, FN


def cross_validation(k, all_messages):
    lambdas_spam = [1]
    lambdas_legit = [1, 10, 100, 1000, 8000]
    alphas = [1, 1e-5]
    return calculate_data(k, all_messages, alphas, lambdas_spam, lambdas_legit, True)


def calculate_data(k, all_messages, alphas, lambdas_spam, lambdas_legit, is_cross_valid):
    best_accuracy = 0
    best_combo_acc = []
    no_legit_as_spam = []
    accuracies, precisions = [], []
    for alpha in alphas:
        for cur_lambda_spam in lambdas_spam:
            for cur_lambda_legit in lambdas_legit:
                if cur_lambda_legit == cur_lambda_spam and cur_lambda_legit != 1:
                    continue
                cur_lambdas = [cur_lambda_spam, cur_lambda_legit]

                legit_as_legit = 0
                legit_as_spam = 0
                spam_as_legit = 0
                spam_as_spam = 0
                for i in range(len(all_messages)):
                    test_messages = all_messages.pop(i)
                    messages = list(msg for msgs in all_messages for msg in msgs)
                    TP, TN, FP, FN = bayes(k, messages, test_messages, alpha, cur_lambdas, [])
                    legit_as_spam += FN
                    legit_as_legit += TP
                    spam_as_legit += FP
                    spam_as_spam += TN
                    all_messages.insert(i, test_messages)

                accuracy = (legit_as_legit + spam_as_spam) / (legit_as_spam + legit_as_legit +
                                                              spam_as_legit + spam_as_spam)
                precision = legit_as_legit / (legit_as_legit + spam_as_legit)
                accuracies.append(accuracy)
                precisions.append(precision)
                print("For lambda_spam:", cur_lambdas[0], "lambda_legit:", cur_lambdas[1], " with alpha =", alpha)
                print("Accuracy:", accuracy, " Precision:", precision)
                print("Legit as spam:", legit_as_spam)
                if is_cross_valid:
                    if legit_as_spam == 0:
                        no_legit_as_spam = [cur_lambdas, alpha, legit_as_spam]

                    if accuracy > best_accuracy:
                        best_combo_acc = [cur_lambdas, alpha, legit_as_spam]
                        best_accuracy = accuracy
                    elif accuracy == best_accuracy and legit_as_spam < best_combo_acc[2]:
                        best_combo_acc = [cur_lambdas, alpha, legit_as_spam]
                        best_accuracy = accuracy

    return no_legit_as_spam, best_combo_acc, best_accuracy, accuracies, precisions


def get_plot_data(k, all_messages, no_legit_as_spam_cmb):
    lambdas_spam = [1]
    lambdas_legit = [i * 500 for i in range(1, int(no_legit_as_spam_cmb[0][1] / 500) + 1)]
    alphas = [no_legit_as_spam_cmb[1]]
    _, _, _, accuracies, precisions = calculate_data(k, all_messages, alphas, lambdas_spam, lambdas_legit,
                                                     False)

    return lambdas_legit, accuracies, precisions


def get_roc_data(k, all_messages, no_legit_as_spam_cmb):
    roc = []
    spams, legits = 0, 0
    for i in range(len(all_messages)):
        test_messages = all_messages.pop(i)
        messages = list(msg for msgs in all_messages for msg in msgs)
        TP, TN, FP, FN = bayes(k, messages, test_messages, no_legit_as_spam_cmb[1], no_legit_as_spam_cmb[0], roc)
        all_messages.insert(i, test_messages)
        spams += TN + FP
        legits += TP + FN

    roc = sorted(roc, key=lambda p_x: p_x[0])
    plus_x, plus_y = 1 / spams, 1 / legits
    roc_x, roc_y = [0], [0]
    cur_x, cur_y = 0, 0
    for i in range(1, len(roc) + 1):
        if roc[len(roc) - i][1] == legit_mark:
            cur_y += plus_y
            roc_x.append(cur_x)
            roc_y.append(cur_y)
        else:
            cur_x += plus_x
            roc_x.append(cur_x)
            roc_y.append(cur_y)

    return roc_x, roc_y


def draw_plot(x, y, x_lable, y_label,title):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(title)
    if title == "ROC curve":
        plt.title(title)
    plt.plot(x, y)
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)


def main():
    all_messages = read_files(1)
    k = 2
    no_legit_as_spam_cmb, best_combo_acc, best_accuracy, _, _ = cross_validation(k, all_messages)
    #no_legit_as_spam_cmb = [[1, 8000], 1e-5, 0]

    print("\nAll legit messages are legit")
    print("For lambda_spam :", no_legit_as_spam_cmb[0][0], ", lambda_legit :", no_legit_as_spam_cmb[0][1],
          " with alpha =", no_legit_as_spam_cmb[1])
    print("Legit as spam:", no_legit_as_spam_cmb[2])

    print("\nBest accuracy")
    print("For lambda_spam :", best_combo_acc[0][0], ", lambda_legit :", best_combo_acc[0][1], " with alpha =",
          best_combo_acc[1])
    print("Accuracy:", best_accuracy)
    print("Legit as spam:", best_combo_acc[2], "\n")

    lambdas, accuracies, precisions = get_plot_data(k, all_messages, no_legit_as_spam_cmb)
    draw_plot(lambdas, accuracies, "lambda", "accuracy","Accuracy")
    draw_plot(lambdas, precisions, "lambda", "precision","Precision")

    roc_x, roc_y = get_roc_data(k, all_messages, no_legit_as_spam_cmb)
    draw_plot(roc_x, roc_y, "FPR", "TPR","ROC curve")
    plt.show()


main()
