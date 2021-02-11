import random


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


def svm(n, y, C, K, k_reg):
    lambdas = [0 for _ in range(n)]
    b = 0

    for _ in range(300):
        for i in range(n):
            err_i = eval_error(i, y, lambdas, K, b)
            r_i = err_i * y[i]
            if match_KKT_conditions(r_i, k_reg, lambdas[i], C):
                j = get_second_ind(i, n)
                err_j = eval_error(j, y, lambdas, K, b)

                to_next, L, H = check_transition_to_next_iter(y[i], y[j], lambdas[i], lambdas[j], C)
                parameter = K[i][i] + K[j][j] - 2 * K[i][j]
                new_lambda_j = min(max((lambdas[j] + y[j] * (err_i - err_j) / parameter), L), H)
                new_lambda_i = lambdas[i] + y[i] * y[j] * (lambdas[j] - new_lambda_j)
                if to_next or parameter < 0 or is_equal(lambdas[j], new_lambda_j):
                    continue

                b1 = err_i + y[i] * (new_lambda_i - lambdas[i]) * K[i][i] + y[j] * (new_lambda_j - lambdas[j]) * K[i][
                    j] + b
                b2 = err_j + y[i] * (new_lambda_i - lambdas[i]) * K[i][j] + y[j] * (new_lambda_j - lambdas[j]) * K[j][
                    j] + b

                b = (b1 + b2) / 2
                lambdas[i] = new_lambda_i
                lambdas[j] = new_lambda_j

    for el in lambdas:
        print(round(el, 10))
    print(-round(b, 10))


def main():
    n = int(input())
    K = []
    y = []
    for _ in range(n):
        elements = list(map(int, input().split()))
        tmp = elements[:-1]
        K.append(tmp)
        y.append(elements[len(elements) - 1])

    C = int(input())
    k_reg = 1e-8
    svm(n, y, C, K, k_reg)


main()
