import math


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


def algo(cur_node, k, print_ind, h_remain):
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

    new_print_ind = algo(left_node, k, print_ind + 1, h_remain - 1)
    return algo(right_node, k, new_print_ind, h_remain - 1)


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


def main():
    m, k, h = map(int, input().split())
    n = int(input())
    train_data = []
    classes = []
    for _ in range(n):
        tmp = list(map(int, input().split()))
        train_data.append(tmp[:-1])
        classes.append(tmp[-1] - 1)

    root = Node(train_data, classes)

    t_size = algo(root, k, 1, h)
    print(t_size - 1)
    print_tree(root)


main()
