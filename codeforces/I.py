import math
from functools import reduce


def tnh(m):
    return [[math.tanh(el) for el in m[i]] for i in range(len(m))]


def rlu(m, alpha):
    return [[el / alpha if el < 0 else el for el in m[i]] for i in range(len(m))]


def mul(m1, m2):
    return [[sum([m1[i][ii] * m2[ii][j] for ii in range(len(m2))]) for j in range(len(m2[0]))] for i in range(len(m1))]


def summ(ms):
    return [[sum([ms[k][j][i] for k in range(len(ms))]) for i in range(len(ms[0][0]))] for j in range(len(ms[0]))]


def had(ms):
    return [[reduce(lambda x, y: x * y, [ms[k][j][i] for k in range(len(ms))]) for i in range(len(ms[0][0]))] for j in
            range(len(ms[0]))]


def d_tnh(m):
    return [[1 / math.pow(math.cosh(el), 2) for el in m[i]] for i in range(len(m))]


def d_rlu(m, alpha):
    return [[1 / alpha if el < 0 else 1 for el in m[i]] for i in range(len(m))]


def transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def main():
    n, m, k = map(int, input().split())

    vertices = []
    d_v = []
    tmps = []
    vs = []
    for i in range(n):
        vertices.append([])
        tmp = list(input().split())
        tmps.append(tmp)
        if tmp[0] == "var":
            vs.append(i)

    for i in vs:
        cur_tmp = []
        for _ in range(int(tmps[i][1])):
            row = list(map(int, input().split()))
            cur_tmp.append(row)
        vertices[i] = cur_tmp

    ind = -1
    for tmp in tmps:
        ind += 1
        key = tmp[0]
        args = list(map(int, tmp[1:]))
        if key == "var":
            d_v.append([[0.0] * len(vertices[ind][0]) for _ in range(len(vertices[ind]))])
            continue
        elif key == "tnh":
            vertices[ind] = tnh(vertices[args[0] - 1])
        elif key == "rlu":
            vertices[ind] = rlu(vertices[args[1] - 1], args[0])
        elif key == "mul":
            vertices[ind] = mul(vertices[args[0] - 1], vertices[args[1] - 1])
        elif key == "sum":
            vertices[ind] = summ([vertices[args[i] - 1] for i in range(1, 1 + args[0])])
        elif key == "had":
            vertices[ind] = had([vertices[args[i] - 1] for i in range(1, 1 + args[0])])

        d_v.append([[0.0 for _ in range(len(vertices[ind][0]))] for _ in range(len(vertices[ind]))] if ind < n - k
                   else [list(map(float, input().split())) for _ in range(len(vertices[ind]))])

    # print(vertices)
    # print(d_v)
    for i in range(n - k, n):
        for s in vertices[i]:
            print(*s)

    for i in range(0, n - m):
        ii = n - 1 - i
        key = tmps[ii][0]
        args = list(map(int, tmps[ii][1:]))
        if key == "var":
            print(5 / 0)
        elif key == "tnh":
            d_v[args[0] - 1] = summ([had([d_tnh(vertices[args[0] - 1]), d_v[ii]]), d_v[args[0] - 1]])
        elif key == "rlu":
            d_v[args[1] - 1] = summ([had([d_rlu(vertices[args[1] - 1], args[0]), d_v[ii]]), d_v[args[1] - 1]])
        elif key == "mul":
            d_v[args[0] - 1] = summ([mul(d_v[ii], transpose(vertices[args[1] - 1])), d_v[args[0] - 1]])
            d_v[args[1] - 1] = summ([mul(transpose(vertices[args[0] - 1]), d_v[ii]), d_v[args[1] - 1]])
        elif key == "sum":
            for arg in args[1:]:
                d_v[arg - 1] = summ([d_v[ii], d_v[arg - 1]])
        elif key == "had":
            indd = 1
            nargs = args[1:]
            for arg in nargs:
                d_v[arg - 1] = summ([had([vertices[e - 1] for e in nargs[:indd - 1] + nargs[indd:]] + [d_v[ii]]),
                                     d_v[arg - 1]])
                indd += 1

    for i in range(m):
        for s in d_v[i]:
            print(*s)


main()


# 6 3 2
# var 1 2
# var 2 3
# var 3 1
# tnh 1
# tnh 4
# tnh 3
# 1 2
# 2 3 4
# 1 2 2
# 2
# 2
# 1
# [[1.0, 2.0]]
# [[2.0, 3.0, 4.0], [1.0, 2.0, 2.0]]
# [[2.0], [2.0], [1.0]]
# [[0.7615941559557649, 0.9640275800758169]]
# [[0.6420149920119997, 0.7460679984455996]]
# [[0.9640275800758169], [0.9640275800758169], [0.7615941559557649]]
# 1 2
# 1
# -1
# 2
# 0.6420149920119997 0.7460679984455996
# 0.9640275800758169
# 0.9640275800758169
# 0.7615941559557649
# 0.24686795258431515 0.06265068459254189
# 0 0 0
# 0 0 0
# 0.07065082485316443
# -0.07065082485316443
# 0.8399486832280523

