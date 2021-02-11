import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def kernel_polynom_uniform(x1, x2, d):
    return np.power(np.dot(np.array(x1), np.array(x2)), d)


def kernel_linear(x1, x2, _):
    return kernel_polynom(x1, x2, 1)


def kernel_polynom(x1, x2, d):
    return np.power(np.dot(np.array(x1), np.array(x2)) + 1, d)


def kernel_radially_basic(x1, x2, beta):
    return np.exp(-beta * np.power(np.linalg.norm(np.array(x1) - np.array(x2)), 2))


def kernel_sigmoid(x1, x2, k):
    c = 0
    return np.tanh(k * np.dot(np.array(x1), np.array(x2)) + c)


dict_kernels = {
    "linear": kernel_linear,
    "polynom_uniform": kernel_polynom_uniform,
    "polynom": kernel_polynom,
    "radially_basic": kernel_radially_basic,
    "sigmoid": kernel_sigmoid,
}


def fun(y, lambdas, k_i, b):
    return sum(y[i] * lambdas[i] * k_i[i] for i in range(len(y))) - b


def eval_error(ind, y, lambdas, K, b):
    return fun(y, lambdas, K[ind], b) - y[ind]


def get_second_ind(first_ind, n):
    second_ind = random.randint(0, n - 1)
    while second_ind == first_ind:
        second_ind = random.randint(0, n - 1)
    return second_ind


def match_KKT_conditions(r_i, k_reg, cur_lambda, C):
    return (r_i < -k_reg and cur_lambda < C) or (r_i > k_reg and cur_lambda > 0)


def is_equal(L, H):
    accurancy = 1e-16
    return abs(L - H) < accurancy


def check_transition_to_next_iter(y_i, y_j, lambda_i, lambda_j, C):
    if y_i != y_j:
        gamma = lambda_j - lambda_i
        L = max(0, gamma)
        H = min(C, C + gamma)
    else:
        gamma = lambda_i + lambda_j
        L = max(0, gamma - C)
        H = min(C, gamma)
    return is_equal(L, H), L, H


def svm(n, x, y, C, k_reg, kernel, kernel_paramentr):
    lambdas = [0 for _ in range(n)]
    b = 0
    K = [[dict_kernels[kernel](x[i], x[j], kernel_paramentr) for j in range(n)] for i in range(n)]

    for _ in range(300):
        for i in range(n):
            err_i = eval_error(i, y, lambdas, K, b)
            r_i = err_i * y[i]
            if match_KKT_conditions(r_i, k_reg, lambdas[i], C):
                j = get_second_ind(i, n)
                err_j = eval_error(j, y, lambdas, K, b)

                to_next, L, H = check_transition_to_next_iter(y[i], y[j], lambdas[i], lambdas[j], C)
                parameter = K[i][i] + K[j][j] - 2 * K[i][j]
                if to_next or parameter <= 0:
                    continue
                new_lambda_j = min(max((lambdas[j] + y[j] * (err_i - err_j) / parameter), L), H)
                new_lambda_i = lambdas[i] + y[i] * y[j] * (lambdas[j] - new_lambda_j)
                if is_equal(lambdas[j], new_lambda_j):
                    continue

                b1 = err_i + y[i] * (new_lambda_i - lambdas[i]) * K[i][i] + y[j] * (new_lambda_j - lambdas[j]) * K[i][
                    j] + b
                b2 = err_j + y[i] * (new_lambda_i - lambdas[i]) * K[i][j] + y[j] * (new_lambda_j - lambdas[j]) * K[j][
                    j] + b

                b = (b1 + b2) / 2
                lambdas[i] = new_lambda_i
                lambdas[j] = new_lambda_j

    return lambdas, b


def draw_plot(dataset, tmp_X, predicted_y, kernel):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title("Kernel " + kernel)
    plt.title("Kernel " + kernel)

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


def predict(y, lambdas, b, x, cur_element, kernel, parametr):
    tmp_n = len(y)
    K = [[dict_kernels[kernel](cur_element[i], x[j], parametr) for j in range(tmp_n)] for i in range(len(cur_element))]
    predicted_y = [np.sign(fun(y, lambdas, K[i], b)) for i in range(len(cur_element))]
    return predicted_y


def cross_validation(n, x, y, k_reg, parameters, constants):
    best_combos = []
    for kernel in dict_kernels:
        min_err = -1
        best_combo = []
        for parameter in parameters:
            for C in constants:
                err = 0
                for i in range(len(x)):
                    cur_elem_x = x.pop(i)
                    cur_elem_y = y.pop(i)
                    lambdas, b = svm(n - 1, x, y, C, k_reg, kernel, parameter)
                    predicted_y = predict(y, lambdas, b, x, [cur_elem_x], kernel, parameter)
                    if predicted_y[0] != cur_elem_y:
                        err += 1
                    x.insert(i, cur_elem_x)
                    y.insert(i, cur_elem_y)

                print("Kernel =", kernel, "with parametr =", parameter, " and C =", C, " ERR =", err)
                if min_err == -1 or err < min_err:
                    best_combo = [kernel, parameter, C]
                    min_err = err
        best_combos.append(best_combo)
    return best_combos


def main():
    dataset = pd.read_csv("chips.csv")
    xy = dataset.values.tolist()
    x = dataset[['x', 'y']].values.tolist()
    n = len(x)
    y = [(1 if dataset[['class']].values[i] == 'P' else -1) for i in range(n)]
    k_reg = 1e-8

    parameters = [1, 2, 3, 10]
    constants = [1, 10, 50]
    #best_combos = cross_validation(n,x, y, k_reg, parameters, constants)
    best_combos = [["linear", 1, 50], ["polynom_uniform", 2, 10], ["polynom", 4, 20], ["radially_basic", 1, 10],
                   ["sigmoid", 1, 50]]

    for best_combo in best_combos:
        kernel, parameter, C = best_combo[0], best_combo[1], best_combo[2]
        print("BEST for Kernel =", kernel, ": parametr =", parameter, " and C =", C)

        lambdas, b = svm(n, x, y, C, k_reg, kernel, parameter)
        train_predicted_y = predict(y, lambdas, b, x, x, kernel, parameter)
        wrong_pred = len(list(filter(lambda lm: lm[0] != lm[1], zip(train_predicted_y, y))))
        print("Wrong predicitions = ", wrong_pred)

        tmp_X = []
        # for i in np.arange(0, 25, 0.5):
        #     for j in np.arange(0, 10, 0.2):
        for i in np.arange(-1, 1.25, 0.05):
            for j in np.arange(-1, 1.25, 0.05):
                tmp_X.append([i, j])
        test_predicted_y = predict(y, lambdas, b, x, tmp_X, kernel, parameter)
        draw_plot(xy, tmp_X, test_predicted_y, kernel)

    plt.show()


main()
