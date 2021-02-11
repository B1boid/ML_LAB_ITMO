import pandas as pd
from matplotlib import pyplot as plt
import math


def manhattan(a, b):
    cur_sum = 0
    for i in range(len(a)):
        cur_sum += abs(a[i] - b[i])
    return cur_sum


def euclidean(a, b):
    cur_sum = 0
    for i in range(len(a)):
        cur_sum += (a[i] - b[i]) ** 2
    return cur_sum ** 0.5


def chebyshev(a, b):
    cur_max = 0
    for i in range(len(a)):
        cur_max = abs(a[i] - b[i]) if cur_max < abs(a[i] - b[i]) else cur_max
    return cur_max


def uniform(u):
    if abs(u) < 1:
        return 0.5
    return 0


def triangular(u):
    if abs(u) < 1:
        return 1 - abs(u)
    return 0


def epanechnikov(u):
    if abs(u) < 1:
        return 3 / 4 * (1 - u ** 2)
    return 0


def quartic(u):
    if abs(u) < 1:
        return 15 / 16 * (1 - u ** 2) ** 2
    return 0


def triweight(u):
    if abs(u) < 1:
        return 35 / 32 * (1 - u ** 2) ** 3
    return 0


def tricube(u):
    if abs(u) < 1:
        return 70 / 81 * (1 - abs(u) ** 3) ** 3
    return 0


def gaussian(u):
    return 1 / (math.sqrt(2 * math.pi)) * math.e ** (-0.5 * u ** 2)


def cosine(u):
    if abs(u) < 1:
        return math.pi / 4 * math.cos(u * math.pi / 2)
    return 0


def logistic(u):
    return 1 / (math.e ** u + 2 + math.e ** (-u))


def sigmoid(u):
    return (2 / math.pi) * (1 / (math.e ** u + math.e ** (-u)))


dict_kernels = {
    "uniform": uniform,
    "triangular": triangular,
    "epanechnikov": epanechnikov,
    "quartic": quartic,
    "triweight": triweight,
    "tricube": tricube,
    "gaussian": gaussian,
    "cosine": cosine,
    "logistic": logistic,
    "sigmoid": sigmoid,
}

dict_dists = {
    "chebyshev": chebyshev,
    "manhattan": manhattan,
    "euclidean": euclidean,
}


def vectorize(dataset):
    for feature in dataset:
        need_vectorize = False
        try:
            int(dataset[feature][0])
        except ValueError:
            need_vectorize = True
        if need_vectorize:
            tmp_dict = {}
            for i in range(len(dataset[feature])):
                if dataset[feature][i] not in tmp_dict:
                    tmp_dict[dataset[feature][i]] = len(tmp_dict)
            dataset[feature] = dataset[feature].map(tmp_dict)
    return dataset


def normalize(dataset, classCol):
    classes = dataset[classCol]
    del dataset[classCol]
    data_norm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    data_norm[classCol] = classes
    dataset[classCol] = classes
    return data_norm


def predict(type_dist, type_kernel, type_window, window, q, cur_dataset):
    distances = []
    for element in cur_dataset:
        distances.append((dict_dists[type_dist](element[:-1], q), element[len(element) - 1]))
    distances.sort()

    h = window if type_window == "fixed" else distances[window][0]
    limit_ans = sum(element[-1] for element in cur_dataset) / (len(cur_dataset))
    if h == 0:
        res = 0
        for i in range(len(distances)):
            if distances[i][0] != 0:
                if i == 0:
                    break
                return res / i
            res += distances[i][1]
        if res != 0:
            return res / len(cur_dataset)
        else:
            return limit_ans
    else:
        y_mul_w = []
        w = []
        for dist in distances:
            k = dict_kernels[type_kernel](dist[0] / h)
            w.append(k)
            y_mul_w.append(k * dist[1])

        return sum(y_mul_w) / sum(w) if sum(w) != 0 else limit_ans


def check_predictions(dataset, class_types, type_dist, type_kernel, type_window, window):
    errors = 0
    err_matrix = [[0] * class_types for _ in range(class_types)]
    for index in range(len(dataset.values)):
        cur_dataset = dataset.drop(index)
        cur_row = dataset.values[index]

        prediction = round(predict(type_dist, type_kernel, type_window, window, q=cur_row.tolist(),
                                   cur_dataset=cur_dataset.values.tolist()))
        err_matrix[int(cur_row.tolist()[-1])][prediction] += 1
        if prediction != cur_row.tolist()[-1]:
            # print(index, "Expected:", cur_row.tolist()[-1], "Prediction:", prediction)
            errors += 1
    print(type_dist, type_kernel, window, "Wrong predictions:", errors)
    return errors, err_matrix


def one_hot(dataset, class_name):
    norm = pd.DataFrame()
    for feature in dataset:
        norm[feature] = dataset[feature]
        if feature == class_name:
            continue
        onehot = pd.get_dummies(dataset[feature])
        del dataset[feature]
        dataset = pd.concat([onehot, dataset], axis=1)
    return dataset,norm


def f_measure(k, matrix):
    sum_row = [sum(row) for row in matrix]
    sum_col = [sum(col) for col in zip(*matrix)]
    sum_all = sum(sum_col)
    micro = 0
    for i in range(k):
        if (sum_col[i] != 0) & (sum_row[i] != 0):
            prec = matrix[i][i] / sum_col[i]
            recall = matrix[i][i] / sum_row[i]
        else:
            prec = 0
            recall = 0
        f = 2 * (prec * recall) / (prec + recall) if prec + recall != 0 else 0
        micro += (sum_row[i] * f) / sum_all
    return micro


def get_plot_data(steps, best_type_dist, best_type_kernel, dataset, class_types):
    f_measures = []
    for width in steps:
        _, error_matrix = check_predictions(dataset=dataset,
                                            class_types=class_types,
                                            type_dist=best_type_dist,
                                            type_kernel=best_type_kernel,
                                            type_window="variable",
                                            window=width)
        f_measures.append(f_measure(class_types, error_matrix))

    print("Max f-measure:",max(f_measures))
    draw_plot(steps, f_measures)


def draw_plot(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Number of neighbours")
    ax.set_ylabel("F-measure")
    plt.show()


def main():
    dataset = pd.read_csv("seed.csv")
    class_name = "Class"
    class_types = 4
    dataset_vectorized = vectorize(dataset)
    dataset_normalized_0 = normalize(dataset_vectorized, class_name)
    dataset_onehot,dataset_normalized = one_hot(dataset_normalized_0, class_name)
    best_combination_simple, best_combination_onehot = [], []
    min_error_simple, min_error_onehot = 1, 1
    for type_dist in dict_dists:
        for type_kernel in dict_kernels:
            for width in [5, 10, 15, 20, 25]:
                cur_errors, _ = check_predictions(dataset=dataset_normalized,
                                                  class_types=class_types,
                                                  type_dist=type_dist,
                                                  type_kernel=type_kernel,
                                                  type_window="variable",
                                                  window=width)
                cur_errors /= len(dataset_normalized)
                if cur_errors < min_error_simple:
                    min_error_simple = cur_errors
                    best_combination_simple = [type_dist, type_kernel, width]

                cur_errors, _ = check_predictions(dataset=dataset_onehot,
                                                  class_types=class_types,
                                                  type_dist=type_dist,
                                                  type_kernel=type_kernel,
                                                  type_window="variable",
                                                  window=width)
                cur_errors /= len(dataset_onehot)
                if cur_errors < min_error_onehot:
                    min_error_onehot = cur_errors
                    best_combination_onehot = [type_dist, type_kernel, width]

    print("Method Simple", min_error_simple, best_combination_simple)
    print("Method OneHot", min_error_onehot, best_combination_onehot)

    #best_combination_onehot = ['euclidean', 'tricube', 15]
    steps = [i * 10 for i in range(int(len(dataset) / 10))]
    get_plot_data(steps, best_combination_onehot[0], best_combination_onehot[1], dataset_onehot, class_types)


main()
