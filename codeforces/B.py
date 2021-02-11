def main():
    k = int(input())
    matrix = []
    for _ in range(k):
        matrix.append(list(map(int, input().split())))
    f_measure(k, matrix)


def f_measure(k, matrix):
    sum_row = [sum(row) for row in matrix]
    sum_col = [sum(col) for col in zip(*matrix)]
    sum_all = sum(sum_col)
    beta = 1
    beta2 = beta ** 2
    macro_f(k, matrix, sum_col, sum_row, sum_all, beta2)
    micro_f(k, matrix, sum_col, sum_row, sum_all, beta2)


def macro_f(k, matrix, sum_col, sum_row, sum_all, beta2):
    prec_w, recall_w = 0, 0

    for i in range(k):
        if sum_col[i] != 0:
            prec_w += matrix[i][i] * sum_row[i] / sum_col[i]
        else:
            prec_w = 0
        recall_w += matrix[i][i]

    prec_w = prec_w / sum_all
    recall_w = recall_w / sum_all
    macro = (1 + beta2) * (prec_w * recall_w) / (beta2 * prec_w + recall_w) if prec_w + recall_w != 0 else 0
    print(macro)


def micro_f(k, matrix, sum_col, sum_row, sum_all, beta2):
    micro = 0
    for i in range(k):
        if (sum_col[i] != 0) & (sum_row[i] != 0):
            prec = matrix[i][i] / sum_col[i]
            recall = matrix[i][i] / sum_row[i]
        else:
            prec = 0
            recall = 0
        f = (1 + beta2) * (prec * recall) / (beta2 * prec + recall) if prec + recall != 0 else 0
        micro += (sum_row[i] * f) / sum_all
    print(micro)


main()
