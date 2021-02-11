import math
import random
import numpy as np
from numpy import linalg as lin
from matplotlib import pyplot as plt


def read_data(f):
    n = int(f.readline())
    data = []
    for _ in range(n):
        cur_str = f.readline()
        data.append([int(x) for x in cur_str.split()])
    return n, data


def parse_data(data):
    x = []
    y = []
    for i in range(len(data)):
        tmp = data[i][:-1]
        tmp.append(1)
        x.append(tmp)
        y.append(data[i][len(data[i]) - 1])
    return x, y


def init_weights(sz):
    max_bound = 1 / (2 * sz)
    return [-max_bound + random.uniform(0, 2 * max_bound) for _ in range(sz)]


def generate_weights(w, T):
    return [cur_w + random.uniform(-T, T) for cur_w in w]


def get_y(x_cur, w):
    return sum(x_cur[i] * w[i] for i in range(len(w)))


def grad_descent(n, m, x, y, steps, k_reg):
    w = init_weights(m + 1)
    for j in range(0, steps):
        cur_ind = random.randint(0, n - 1)
        dist = get_y(x[cur_ind], w) - y[cur_ind]
        derivative = []
        for i in range(m + 1):
            derivative.append(2 * dist * x[cur_ind][i])
        dx = get_y(x[cur_ind], derivative)
        if dx != 0:
            h = dist / dx
        else:
            continue
        for i in range(m + 1):
            w[i] = w[i] * (1 - h * k_reg) - h * derivative[i]
    return w


def find_k_reg(n, m, x_train, y_train):
    min_err = -1
    best_k_reg = 0
    for k_reg in range(0, 100, 10):
        err = 0
        for i in range(len(x_train)):
            cur_elem_x = x_train.pop(i)
            cur_elem_y = y_train.pop(i)
            cur_grad = grad_descent(n - 1, m, x_train, y_train, 100, k_reg)
            err += (cur_elem_y - get_y(cur_elem_x, cur_grad)) ** 2
            x_train.insert(i, cur_elem_x)
            y_train.insert(i, cur_elem_y)

        if min_err == -1 or err < min_err:
            best_k_reg = k_reg
            min_err = err
    return best_k_reg


def find_t_start(m, x_train, y_train):
    min_err = -1
    best_T_start = 0
    for t in range(0, 100, 10):
        err = 0
        for i in range(len(x_train)):
            cur_elem_x = x_train.pop(i)
            cur_elem_y = y_train.pop(i)
            cur_w = simulated_annealing(m, np.array(x_train), y_train, 0.001, t)
            err += (cur_elem_y - get_y(cur_elem_x, cur_w)) ** 2
            x_train.insert(i, cur_elem_x)
            y_train.insert(i, cur_elem_y)
        if min_err == -1 or err < min_err:
            best_T_start = t
            min_err = err
    return best_T_start


def least_squares(x, y):
    pseudoinverse = lin.pinv(x)
    return pseudoinverse.dot(y)


def nrmse(X, y, w):
    y_pred = X.dot(w)
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))
    return rmse / (np.max(y) - np.min(y))


def draw_plot(file_name, x, y1, y2, y3, y4, y5, y6):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Data ' + file_name)
    plt.plot(x, y1, label='grad_descent test')
    plt.plot(x, y2, label='grad_descent train')
    plt.plot(x, y3, label='least_squares test')
    plt.plot(x, y4, label='least_squares train')
    plt.plot(x, y5, label='sim_ann test')
    plt.plot(x, y6, label='sim_ann train')
    ax.set_xlabel("iterations")
    ax.set_ylabel("nrmse")
    plt.legend()
    plt.show()


def simulated_annealing(m, train_X, y_train, limit, T_start):
    w = init_weights(m + 1)
    T = T_start
    cur_energy = nrmse(train_X, y_train, w)
    i = 2
    while T > limit:
        w_candidate = generate_weights(w, 1 / T)
        energy_candidate = nrmse(train_X, y_train, w_candidate)
        if energy_candidate < cur_energy:
            cur_energy = energy_candidate
            w = w_candidate
        else:
            p = math.exp(-(energy_candidate - cur_energy) / T)
            if p >= random.uniform(0, 1):
                cur_energy = energy_candidate
                w = w_candidate
        T = T_start / i
        i += 1
    return w


def main():
    file_name = '3.txt'
    f = open(file_name)
    m = int(f.readline())
    n, data_train = read_data(f)
    k, data_test = read_data(f)
    f.close()
    x_train, y_train = parse_data(data_train)
    x_test, y_test = parse_data(data_test)
    train_X = np.array(x_train)
    test_X = np.array(x_test)

    w_least_squares = least_squares(x_train, y_train)
    print("W(least_squares) =", list(w_least_squares))

    err_mnk_test = nrmse(test_X, y_test, w_least_squares)
    print("Error(least_squares) for test data =", "%.10f" % err_mnk_test)
    err_mnk_train = nrmse(train_X, y_train, w_least_squares)
    print("Error(least_squares) for train data =", "%.10f" % err_mnk_train)

    k_reg = find_k_reg(n,m,x_train,y_train)
    #k_reg = 40
    print("Best regularization hyperparameter =", k_reg)

    res_grad_100 = grad_descent(n, m, x_train, y_train, 100, k_reg)
    print("W(gradient descent 100 iterations)", res_grad_100)
    res_grad_1000 = grad_descent(n, m, x_train, y_train, 1000, k_reg)
    print("W(gradient descent 1000 iterations)", res_grad_1000)
    res_grad_10000 = grad_descent(n, m, x_train, y_train, 10000, k_reg)
    print("W(gradient descent 10000 iterations)", res_grad_10000)

    err_grad_100 = nrmse(test_X, y_test, res_grad_100)
    print("Error(gradient descent 100 iterations)", "%.10f" % err_grad_100)
    err_grad_1000 = nrmse(test_X, y_test, res_grad_1000)
    print("Error(gradient descent 1000 iterations)", "%.10f" % err_grad_1000)
    err_grad_10000 = nrmse(test_X, y_test, res_grad_10000)
    print("Error(gradient descent 10000 iterations)", "%.10f" % err_grad_10000)

    steps = [i * 10 for i in range(0, 100)]
    errs_mnk_test = np.repeat(err_mnk_test, len(steps))
    errs_mnk_train = np.repeat(err_mnk_train, len(steps))
    w_grads = []
    err_grads_test = []
    err_grads_train = []
    for step in steps:
        cur_grad = grad_descent(n, m, x_train, y_train, step, k_reg)
        cur_err_grad_test = nrmse(test_X, y_test, cur_grad)
        cur_err_grad_train = nrmse(train_X, y_train, cur_grad)
        w_grads.append(cur_grad)
        err_grads_test.append(cur_err_grad_test)
        err_grads_train.append(cur_err_grad_train)

    #T_best = find_t_start(m,x_train,y_train)
    T_best = 60
    steps2 = [(100 - i) / 1000 for i in range(0, 100)]
    err_sim_ann_test = []
    err_sim_ann_train = []
    for limit in steps2:
        cur_w = simulated_annealing(m, train_X, y_train, limit, T_best)
        cur_err_test = nrmse(test_X, y_test, cur_w)
        cur_err_train = nrmse(train_X, y_train, cur_w)
        err_sim_ann_test.append(cur_err_test)
        err_sim_ann_train.append(cur_err_train)

    draw_plot(file_name, steps, err_grads_test, err_grads_train, errs_mnk_test, errs_mnk_train,
              err_sim_ann_test, err_sim_ann_train)


main()
