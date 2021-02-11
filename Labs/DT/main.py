import math
import random
import numpy as np
from matplotlib import pyplot as plt


class Node:
    def __init__(self, cur_data=None, cur_data_classes=None):
        self.left_node = None
        self.right_node = None
        self.predicat = None
        self.predicat_ind = None
        self.cur_class = None
        self.print_ind = 0
        self.data = cur_data
        self.data_classes = cur_data_classes


def find_probabilities(classes, k):
    probabilities = [0 for _ in range(k)]
    for el in classes:
        probabilities[el] += 1
    return probabilities


def find_entropy(classes, k):
    probabilities = find_probabilities(classes, k)
    s = 0
    for el in probabilities:
        p = el / len(classes)
        if p != 0:
            s += p * math.log2(p)
    return s


def find_predicats(data):
    predicats = [sum(col) / len(data) for col in zip(*data)]
    return predicats


def get_parts_by_predicat(predicat, ind, data, classes):
    l_part, l_part_y, r_part, r_part_y = [], [], [], []
    for i in range(len(data)):
        if data[i][ind] < predicat:
            l_part.append(data[i])
            l_part_y.append(classes[i])
        else:
            r_part.append(data[i])
            r_part_y.append(classes[i])
    return [l_part, l_part_y], [r_part, r_part_y]


def get_nearest_class(classes):
    return max(set(classes), key=classes.count)


def build_tree(cur_node, k, print_ind, h_remain):
    entropy = find_entropy(cur_node.data_classes, k)
    cur_node.print_ind = print_ind
    if entropy == 0:
        cur_node.cur_class = cur_node.data_classes[0]
        return print_ind + 1
    if h_remain == 0:
        cur_node.cur_class = get_nearest_class(cur_node.data_classes)
        return print_ind + 1
    predicats = find_predicats(cur_node.data)
    best_combo = []
    predicat_ind = 0
    for predicat in predicats:
        cur_left_part, cur_right_part = get_parts_by_predicat(predicat, predicat_ind, cur_node.data,
                                                              cur_node.data_classes)
        entropy_left = find_entropy(cur_left_part[1], k)
        entropy_right = find_entropy(cur_right_part[1], k)
        delta_s = abs(entropy - (entropy_left + entropy_right) / 2)
        if len(best_combo) == 0 or delta_s > best_combo[0]:
            best_combo = [delta_s, predicat, predicat_ind, cur_left_part, cur_right_part]
        predicat_ind += 1

    cur_node.predicat = best_combo[1]
    cur_node.predicat_ind = best_combo[2]

    left_node = Node(best_combo[3][0], best_combo[3][1])
    right_node = Node(best_combo[4][0], best_combo[4][1])
    cur_node.left_node = left_node
    cur_node.right_node = right_node

    new_print_ind = build_tree(left_node, k, print_ind + 1, h_remain - 1)
    return build_tree(right_node, k, new_print_ind, h_remain - 1)


def print_tree(cur):
    # print(cur.data_classes,cur.data,cur.cur_class,cur.predicat,cur.predicat_ind,cur.print_ind)
    if cur.cur_class is not None:
        print("C", cur.cur_class + 1)
    else:
        print("Q", cur.predicat_ind + 1, cur.predicat, cur.left_node.print_ind, cur.right_node.print_ind)

    if cur.left_node is not None:
        print_tree(cur.left_node)

    if cur.right_node is not None:
        print_tree(cur.right_node)


def read_data():
    files_number = 21
    cur_dir = "DT_txt/"
    all_data_train = []
    all_data_test = []
    for i in range(files_number):
        file_train = open(cur_dir + str(i + 1).zfill(2) + "_train.txt")
        all_data_train.append(read_file(file_train))
        file_train.close()
        file_test = open(cur_dir + str(i + 1).zfill(2) + "_test.txt")
        all_data_test.append(read_file(file_test))
        file_test.close()
    return all_data_train, all_data_test


def read_file(file):
    first_str = file.readline().strip().split()
    m, k = int(first_str[0]), int(first_str[1])
    n = int(file.readline().strip())
    data = []
    classes = []
    for _ in range(n):
        tmp = list(map(int, file.readline().strip().split()))
        data.append(tmp[:-1])
        classes.append(tmp[-1] - 1)
    return [m, k, n, data, classes]


def get_element_class(trees, element):
    votes = []
    for root in trees:
        node = root
        while node.cur_class is None:
            if element[node.predicat_ind] < node.predicat:
                node = node.left_node
            else:
                node = node.right_node
        votes.append(node.cur_class)
    return get_nearest_class(votes)


def get_error(trees, data_test, classes_test):
    err = 0
    for i in range(len(data_test)):
        err = err + 1 if get_element_class(trees, data_test[i]) != classes_test[i] else err
    return err


def get_best_heights(all_data_train, all_data_test):
    all_combos = []
    for data_ind in range(len(all_data_train)):
        m, k, n = all_data_train[data_ind][0], all_data_train[data_ind][1], all_data_train[data_ind][2]

        heights = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100]
        min_error = -1
        best_height = -1
        for h in heights:
            root = Node(all_data_train[data_ind][3], all_data_train[data_ind][4])
            t_size = build_tree(root, k, 1, h)
            # print(t_size - 1)
            # print_tree(root)
            data_test, classes_test = all_data_test[data_ind][3], all_data_test[data_ind][4]
            cur_error = get_error([root], data_test, classes_test)

            print("Height:", h, " Error =", cur_error)
            if min_error == -1 or cur_error < min_error:
                min_error = cur_error
                best_height = h

        all_combos.append([best_height, data_ind, (n - min_error) / n])
        print("Best height(data" + str(data_ind + 1) + ") =", best_height, " with accuracy:", (n-min_error)/n)
    all_combos.sort(key=lambda x: x[0])
    return all_combos


def draw_plot(x, y1, y2, title):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(title)
    plt.title(title)
    plt.plot(x, y1, label='Тестовый набор')
    plt.plot(x, y2, label='Тренировочный набор')
    ax.set_xlabel("Высота")
    ax.set_ylabel("Точность")
    plt.legend()


def get_plot_data(title, full_train_data, full_test_data):
    heights = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
    k, n = full_train_data[1], full_train_data[2]
    data_train, classes_train = full_train_data[3], full_train_data[4]
    data_test, classes_test = full_test_data[3], full_test_data[4]
    acc_train = []
    acc_test = []
    print(title)
    for h in heights:
        root = Node(data_train, classes_train)
        build_tree(root, k, 1, h)

        cur_error = get_error([root], data_train, classes_train)
        acc_train.append((n - cur_error) / n)
        cur_error = get_error([root], data_test, classes_test)
        acc_test.append((n - cur_error) / n)
        print("For height:", h, " accuracy =", (n - cur_error) / n)

    draw_plot(heights, acc_test, acc_train, title)


def random_elements_and_features(random_kf, n, m, data_train, classes_train):
    data = []
    classes = []
    dict_m = {}
    for i in range(random_kf[1]):
        dict_m[random.randint(0, m - 1)] = 1
    for _ in range(random_kf[0]):
        i = random.randint(0, n - 1)

        data.append(data_train[i])
        classes.append(classes_train[i])

    dt = np.delete(data, list(map(int, dict_m.keys())), axis=1)
    return dt.tolist(), classes


def build_random_forest(all_data_train, all_data_test, all_combos):
    all_combos.sort(key=lambda x: x[1])
    forest_acc = []
    for ind in range(len(all_data_train)):
        m, k, n = all_data_train[ind][0], all_data_train[ind][1], all_data_train[ind][2]
        data_train, classes_train = all_data_train[ind][3], all_data_train[ind][4]
        data_test, classes_test = all_data_test[ind][3], all_data_test[ind][4]

        random_kfs = [[random.randint(int(0.5 * n), int(0.9 * n)), random.randint(int(0.1 * m), int(0.2 * m))],
                      [random.randint(int(0.5 * n), int(0.9 * n)), random.randint(int(0.1 * m), int(0.2 * m))],
                      [random.randint(int(0.4 * n), int(0.7 * n)), random.randint(int(0.1 * m), int(0.2 * m))],
                      [random.randint(int(0.4 * n), int(0.8 * n)), random.randint(int(0.3 * m), int(0.3 * m))],
                      [random.randint(int(0.5 * n), int(0.7 * n)), random.randint(int(0.2 * m), int(0.4 * m))]]

        trees = []
        for random_kf in random_kfs:
            new_data_train, new_data_classes = random_elements_and_features(random_kf, n, m, data_train, classes_train)
            cur_root = Node(new_data_train, new_data_classes)
            build_tree(cur_root, k, 1, all_combos[ind][0])
            trees.append(cur_root)
        cur_error = get_error(trees, data_test, classes_test)
        forest_acc.append((n - cur_error) / n)
    return forest_acc


def draw_comparison(all_combos, y2):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title("Сравнение дерева с лесом")
    plt.title("Сравнение дерева с лесом")
    x = np.array([i for i in range(1, 22)])
    y1 = [el[2] for el in all_combos]
    offset = 0.15
    plt.bar(x - offset, y1, label='Дерево', width=offset)
    plt.bar(x, y2, label='Лес', width=offset)
    ax.set_xlabel("Номер набора")
    ax.set_ylabel("Точность")
    plt.legend()


def main():
    all_data_train, all_data_test = read_data()

    all_combos = get_best_heights(all_data_train, all_data_test)
    min_height_combo = all_combos[0]
    mid_height_combo = all_combos[int(len(all_combos) / 2)]
    max_height_combo = all_combos[-1]
    # min_height_combo = [3, 4]
    # mid_height_combo = [7, 15]
    # max_height_combo = [10, 0]
    print("Min height =", min_height_combo[0], " for data" + str(min_height_combo[1] + 1))
    print("Mid height =", mid_height_combo[0], " for data" + str(mid_height_combo[1] + 1))
    print("Max height =", max_height_combo[0], " for data" + str(max_height_combo[1] + 1))

    get_plot_data("Минимальная высота", all_data_train[min_height_combo[1]], all_data_test[min_height_combo[1]])
    get_plot_data("Средняя высота", all_data_train[mid_height_combo[1]], all_data_test[mid_height_combo[1]])
    get_plot_data("Максимальная высота", all_data_train[max_height_combo[1]], all_data_test[max_height_combo[1]])

    forest_acc = build_random_forest(all_data_train, all_data_test, all_combos)
    draw_comparison(all_combos, forest_acc)
    plt.show()


main()
