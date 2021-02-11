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
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev
}


def main():
    n, m = map(int, input().split())
    d = []
    for _ in range(n):
        d.append(list(map(int, input().split())))
    q = list(map(int, input().split()))
    type_dist = input()
    type_kernel = input()
    type_window = input()
    window = int(input())

    distances = []
    for element in d:
        distances.append((dict_dists[type_dist](element[:-1], q), element[len(element) - 1]))
    distances.sort()

    h = window if type_window == "fixed" else distances[window][0]
    limit_ans = sum(element[-1] for element in d) / n
    if h == 0:
        res = 0
        for i in range(len(distances)):
            if distances[i][0] != 0:
                if i == 0:
                    break
                print(res / i)
                return
            res += distances[i][1]
        if res != 0:
            print(res / n)
        else:
            print(limit_ans)
    else:
        y_mul_w = []
        w = []
        for dist in distances:
            k = dict_kernels[type_kernel](dist[0] / h)
            w.append(k)
            y_mul_w.append(k * dist[1])
        print(sum(y_mul_w) / sum(w) if sum(w) != 0 else limit_ans)


main()
