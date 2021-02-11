import pandas as pd
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

isChips = True


def ada_boost(dataset, x, y, n):
    weights = [1 / n for _ in range(n)]

    draw_iters = [1, 2, 3, 5, 8, 13, 21, 34, 55,200]
    iterations = 200
    list_iters, accurancies = [], []
    trees = 20
    classifiers, alphas, all_predictions = [], [], []
    for itr in range(1, iterations + 1):
        list_iters.append(itr)
        min_eps = -1
        best_classifier, best_predictions = None, None
        for _ in range(trees):
            classifier = DecisionTreeClassifier(max_depth=2)
            random_ind = [random.randint(0, n - 1) for _ in range(int(np.sqrt(n)))]
            new_x, new_y = [], []
            for i in random_ind:
                new_x.append(x[i])
                new_y.append(y[i])

            classifier.fit(new_x, new_y)
            predictions = classifier.predict(x)
            eps = sum([weights[i] if y[i] != predictions[i] else 0 for i in range(n)])
            if min_eps == -1 or eps < min_eps:
                min_eps = eps
                best_classifier = classifier
                best_predictions = predictions

        alpha = np.log((1 - min_eps) / min_eps)
        alphas.append(alpha)
        classifiers.append(best_classifier)
        all_predictions.append(best_predictions)

        z_norm = 0
        for i in range(n):
            weights[i] *= np.exp(-alpha * y[i] * best_predictions[i])
            z_norm += weights[i]
        weights = [weight / z_norm for weight in weights]

        cur_predictions = get_predictions(x,classifiers,alphas)
        wrong_pred = len(list(filter(lambda lm: lm[0] != lm[1], zip(cur_predictions, y))))
        print("Iteration-"+str(itr)+" Wrong predicitions = ", wrong_pred)
        accurancies.append((n-wrong_pred)/n)

        if itr in draw_iters:
            predict_plot_fragmentation(dataset, classifiers, alphas, itr)

    return list_iters,accurancies


def get_predictions(cur_x,classifiers,alphas):
    predictions = []
    for el in cur_x:
        cur_pred = 0
        for i in range(len(classifiers)):
            ys = classifiers[i].predict([el])
            cur_pred += ys[0] * alphas[i]
        predictions.append(1 if cur_pred > 0 else -1)
    return predictions


def predict_plot_fragmentation(dataset, classifiers, alphas, itr):
    tmp_X = []
    if isChips:
        for i in np.arange(-1, 1.25, 0.05):
            for j in np.arange(-1, 1.25, 0.05):
                tmp_X.append([i, j])
    else:
        for i in np.arange(0, 25, 0.5):
            for j in np.arange(0, 10, 0.2):
                tmp_X.append([i, j])

    plot_predictions = get_predictions(tmp_X,classifiers,alphas)
    draw_plot(dataset, tmp_X, plot_predictions, itr)


def draw_plot(dataset, tmp_X, predicted_y, title):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title("Iteration " + str(title))
    plt.title("Iteration " + str(title))

    x_pos, y_pos, x_neg, y_neg = [], [], [], []
    for i in range(len(tmp_X)):
        if predicted_y[i] == 1:
            x_pos.append(tmp_X[i][0])
            y_pos.append(tmp_X[i][1])
        else:
            x_neg.append(tmp_X[i][0])
            y_neg.append(tmp_X[i][1])

    ax.scatter(x_pos, y_pos, c='r', alpha=0.3)
    ax.scatter(x_neg, y_neg, c='b', alpha=0.3)

    x_pos, y_pos, x_neg, y_neg = [], [], [], []
    for element in dataset:
        if element[2] == 'P':
            x_pos.append(element[0])
            y_pos.append(element[1])
        else:
            x_neg.append(element[0])
            y_neg.append(element[1])

    ax.scatter(x_pos, y_pos, c='r')
    ax.scatter(x_neg, y_neg, c='b')

    ax.set_xlabel("x")
    ax.set_ylabel("y")


def draw_accuracy_plot(x,y):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title("Зависимость точности от итерации")
    plt.title("Зависимость точности от итерации")
    plt.plot(x, y)
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Точность")


def main():
    file_chips = "chips.csv"
    file_geyser = "geyser.csv"
    dataset = pd.read_csv(file_chips if isChips else file_geyser)
    xy = dataset.values.tolist()
    x = dataset[['x', 'y']].values.tolist()
    n = len(x)
    y = [(1 if dataset[['class']].values[i] == 'P' else -1) for i in range(n)]
    list_iters,accurancies = ada_boost(xy, x, y, n)
    draw_accuracy_plot(list_iters,accurancies)
    plt.show()


main()
